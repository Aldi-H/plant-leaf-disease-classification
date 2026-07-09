package main

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// SplitConfig holds the dataset splitting configuration.
type SplitConfig struct {
	// SourceDir is the root directory containing class subdirectories,
	// e.g., "plantvillage dataset/segmented/"
	SourceDir string

	// OutputDir is where train/, validation/, test/ will be created,
	// e.g., "plantvillage dataset/dataset-split/"
	OutputDir string

	// TestRatio is the fraction of data reserved for testing (first split).
	// Default: 0.3 (30%)
	TestRatio float64

	// ValRatio is the fraction of the remaining train data reserved for validation (second split).
	// Default: 0.2 (20% of the 70% → 14% overall)
	ValRatio float64

	// Seed for reproducible shuffling (matches random_state=42 in Python).
	Seed int64
}

func printProgress(classIdx, totalClasses, filesDone, filesTotal int, className string, startTime time.Time) {
	const barWidth = 20

	var pct float64
	if filesTotal > 0 {
		pct = float64(filesDone) / float64(filesTotal) * 100
	}

	filled := int(float64(barWidth) * float64(filesDone) / float64(filesTotal))
	if filled > barWidth {
		filled = barWidth
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

	elapsed := time.Since(startTime)
	var eta time.Duration
	var speed float64
	if filesDone > 0 {
		speed = float64(filesDone) / elapsed.Seconds()
		remaining := filesTotal - filesDone
		eta = time.Duration(float64(remaining)/speed) * time.Second
	}

	displayName := className
	if len(displayName) > 15 {
		displayName = displayName[:12] + "..."
	}

	fmt.Printf("\r\033[2K[%d/%d] %3.0f%% |%s| %d/%d [%s<%s, %.0f f/s] %s",
		classIdx, totalClasses,
		pct, bar, filesDone, filesTotal,
		formatDuration(elapsed), formatDuration(eta), speed,
		displayName)
}

func formatDuration(d time.Duration) string {
	totalSec := int(d.Seconds())
	min := totalSec / 60
	sec := totalSec % 60
	return fmt.Sprintf("%02d:%02d", min, sec)
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <source_dir> <output_dir> [seed]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s './plantvillage dataset/segmented' './plantvillage dataset/dataset-split'\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nSplits dataset into train (56%%), validation (14%%), and test (30%%).\n")
		os.Exit(1)
	}

	cfg := SplitConfig{
		SourceDir: os.Args[1],
		OutputDir: os.Args[2],
		TestRatio: 0.3,
		ValRatio:  0.2,
		Seed:      42,
	}

	if len(os.Args) >= 4 {
		var seed int64
		if _, err := fmt.Sscanf(os.Args[3], "%d", &seed); err == nil {
			cfg.Seed = seed
		}
	}

	if err := splitDataset(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func splitDataset(cfg SplitConfig) error {
	classNames, err := listSubdirs(cfg.SourceDir)
	if err != nil {
		return fmt.Errorf("reading source directory: %w", err)
	}

	if len(classNames) == 0 {
		return fmt.Errorf("no class subdirectories found in %q", cfg.SourceDir)
	}

	fmt.Printf("Found %d classes in %s\n", len(classNames), cfg.SourceDir)

	rng := rand.New(rand.NewSource(cfg.Seed))

	totalTrain, totalVal, totalTest := 0, 0, 0

	startTime := time.Now()
	totalFiles := 0
	copiedFiles := 0

	for _, className := range classNames {
		classDir := filepath.Join(cfg.SourceDir, className)
		files, err := listFiles(classDir)
		if err != nil {
			return fmt.Errorf("listing files in %s: %w", classDir, err)
		}
		totalFiles += len(files)
	}

	for i, className := range classNames {
		classDir := filepath.Join(cfg.SourceDir, className)

		images, err := listFiles(classDir)
		if err != nil {
			return fmt.Errorf("listing files in %s: %w", classDir, err)
		}

		if len(images) == 0 {
			printProgress(i+1, len(classNames), copiedFiles, totalFiles, className, startTime)
			continue
		}

		rng.Shuffle(len(images), func(i, j int) {
			images[i], images[j] = images[j], images[i]
		})

		// Split datasets into train and test with ratio 70:30
		testCount := int(float64(len(images)) * cfg.TestRatio)
		trainValImages := images[:len(images)-testCount]
		testImages := images[len(images)-testCount:]

		// Split datasets into train and validation (80:20 → 56:14 overall)
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
				src := filepath.Join(classDir, imgName)
				dst := filepath.Join(destDir, imgName)
				if err := copyFile(src, dst); err != nil {
					return fmt.Errorf("copying %s → %s: %w", src, dst, err)
				}
				copiedFiles++

				if copiedFiles%50 == 0 {
					printProgress(i+1, len(classNames), copiedFiles, totalFiles, className, startTime)
				}
			}
		}

		totalTrain += len(trainImages)
		totalVal += len(valImages)
		totalTest += len(testImages)

		printProgress(i+1, len(classNames), copiedFiles, totalFiles, className, startTime)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("─", 60))
	fmt.Printf("Summary:\n")
	fmt.Printf("  Train:      %d images\n", totalTrain)
	fmt.Printf("  Validation: %d images\n", totalVal)
	fmt.Printf("  Test:       %d images\n", totalTest)
	fmt.Printf("  Total:      %d images\n", totalTrain+totalVal+totalTest)
	fmt.Printf("  Output:     %s\n", cfg.OutputDir)

	elapsed := time.Since(startTime)
	fmt.Printf("  Elapsed:    %s\n", formatDuration(elapsed))

	return nil
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
