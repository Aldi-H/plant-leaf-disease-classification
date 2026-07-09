package main

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// SplitConfig holds the dataset splitting configuration.
type SplitConfig struct {
	SourceDir string
	OutputDir string
	TestRatio float64
	ValRatio  float64
	Seed      int64
	Workers   int
}

type copyJob struct {
	src string
	dst string
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <source_dir> <output_dir> [seed] [workers]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s './plantvillage dataset/segmented' './plantvillage dataset/dataset-split'\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nSplits dataset into train (56%%), validation (14%%), and test (30%%) using a goroutine worker pool.\n")
		os.Exit(1)
	}

	cfg := SplitConfig{
		SourceDir: os.Args[1],
		OutputDir: os.Args[2],
		TestRatio: 0.3,
		ValRatio:  0.2,
		Seed:      42,
		Workers:   runtime.NumCPU() * 8,
	}

	if len(os.Args) >= 4 {
		var seed int64
		if _, err := fmt.Sscanf(os.Args[3], "%d", &seed); err == nil {
			cfg.Seed = seed
		}
	}
	if len(os.Args) >= 5 {
		var workers int
		if _, err := fmt.Sscanf(os.Args[4], "%d", &workers); err == nil && workers > 0 {
			cfg.Workers = workers
		}
	}

	if err := splitDatasetParallel(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func splitDatasetParallel(cfg SplitConfig) error {
	classNames, err := listSubdirs(cfg.SourceDir)
	if err != nil {
		return fmt.Errorf("reading source directory: %w", err)
	}
	if len(classNames) == 0 {
		return fmt.Errorf("no class subdirectories found in %q", cfg.SourceDir)
	}

	fmt.Printf("Found %d classes in %s\n", len(classNames), cfg.SourceDir)

	rng := rand.New(rand.NewSource(cfg.Seed))

	// Listing, shuffling, and directory creation are cheap and stay sequential;
	// only the file copies (the actual bottleneck) get fanned out to workers.
	var jobs []copyJob
	totalTrain, totalVal, totalTest := 0, 0, 0

	for _, className := range classNames {
		classDir := filepath.Join(cfg.SourceDir, className)

		images, err := listFiles(classDir)
		if err != nil {
			return fmt.Errorf("listing files in %s: %w", classDir, err)
		}
		if len(images) == 0 {
			continue
		}

		rng.Shuffle(len(images), func(i, j int) {
			images[i], images[j] = images[j], images[i]
		})

		testCount := int(float64(len(images)) * cfg.TestRatio)
		trainValImages := images[:len(images)-testCount]
		testImages := images[len(images)-testCount:]

		valCount := int(float64(len(trainValImages)) * cfg.ValRatio)
		trainImages := trainValImages[:len(trainValImages)-valCount]
		valImages := trainValImages[len(trainValImages)-valCount:]

		splits := map[string][]string{
			"train":      trainImages,
			"validation": valImages,
			"test":       testImages,
		}

		for splitName, splitImages := range splits {
			destDir := filepath.Join(cfg.OutputDir, splitName, className)
			if err := os.MkdirAll(destDir, 0o755); err != nil {
				return fmt.Errorf("creating directory %s: %w", destDir, err)
			}
			for _, imgName := range splitImages {
				jobs = append(jobs, copyJob{
					src: filepath.Join(classDir, imgName),
					dst: filepath.Join(destDir, imgName),
				})
			}
		}

		totalTrain += len(trainImages)
		totalVal += len(valImages)
		totalTest += len(testImages)
	}

	totalFiles := len(jobs)
	fmt.Printf("Copying %d files with %d worker goroutines...\n", totalFiles, cfg.Workers)

	startTime := time.Now()
	var copiedFiles int64

	var errMu sync.Mutex
	var copyErrors []string

	jobCh := make(chan copyJob, cfg.Workers*4)

	var workerWG sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		workerWG.Add(1)
		go func() {
			defer workerWG.Done()
			for job := range jobCh {
				if err := copyFile(job.src, job.dst); err != nil {
					errMu.Lock()
					copyErrors = append(copyErrors, fmt.Sprintf("%s -> %s: %v", job.src, job.dst, err))
					errMu.Unlock()
					continue
				}
				atomic.AddInt64(&copiedFiles, 1)
			}
		}()
	}

	stopProgress := make(chan struct{})
	progressDone := make(chan struct{})
	go func() {
		defer close(progressDone)
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				printProgress(atomic.LoadInt64(&copiedFiles), int64(totalFiles), startTime)
			case <-stopProgress:
				return
			}
		}
	}()

	for _, job := range jobs {
		jobCh <- job
	}
	close(jobCh)
	workerWG.Wait()
	close(stopProgress)
	<-progressDone
	printProgress(atomic.LoadInt64(&copiedFiles), int64(totalFiles), startTime)
	fmt.Println()

	if len(copyErrors) > 0 {
		for i, e := range copyErrors {
			if i >= 5 {
				fmt.Fprintf(os.Stderr, "...and %d more errors\n", len(copyErrors)-5)
				break
			}
			fmt.Fprintf(os.Stderr, "copy error: %s\n", e)
		}
		return fmt.Errorf("%d file(s) failed to copy", len(copyErrors))
	}

	fmt.Println(strings.Repeat("─", 60))
	fmt.Printf("Summary:\n")
	fmt.Printf("  Train:      %d images\n", totalTrain)
	fmt.Printf("  Validation: %d images\n", totalVal)
	fmt.Printf("  Test:       %d images\n", totalTest)
	fmt.Printf("  Total:      %d images\n", totalTrain+totalVal+totalTest)
	fmt.Printf("  Output:     %s\n", cfg.OutputDir)
	fmt.Printf("  Workers:    %d\n", cfg.Workers)
	fmt.Printf("  Elapsed:    %s\n", formatDuration(time.Since(startTime)))

	return nil
}

func printProgress(done, total int64, startTime time.Time) {
	const barWidth = 30

	var pct float64
	if total > 0 {
		pct = float64(done) / float64(total) * 100
	}

	filled := int(float64(barWidth) * float64(done) / float64(total))
	if filled > barWidth {
		filled = barWidth
	}
	if filled < 0 {
		filled = 0
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

	elapsed := time.Since(startTime)
	var eta time.Duration
	var speed float64
	if done > 0 {
		speed = float64(done) / elapsed.Seconds()
		remaining := total - done
		if speed > 0 {
			eta = time.Duration(float64(remaining)/speed) * time.Second
		}
	}

	fmt.Printf("\r\033[2K%3.0f%% |%s| %d/%d [%s<%s, %.0f f/s]",
		pct, bar, done, total,
		formatDuration(elapsed), formatDuration(eta), speed)
}

func formatDuration(d time.Duration) string {
	totalSec := int(d.Seconds())
	min := totalSec / 60
	sec := totalSec % 60
	return fmt.Sprintf("%02d:%02d", min, sec)
}

func listSubdirs(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, e.Name())
		}
	}
	sort.Strings(dirs)
	return dirs, nil
}

func listFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var files []string
	for _, e := range entries {
		if !e.IsDir() {
			files = append(files, e.Name())
		}
	}
	sort.Strings(files)
	return files, nil
}

func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	srcInfo, err := srcFile.Stat()
	if err != nil {
		return err
	}

	dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, srcInfo.Mode())
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	return err
}
