use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

struct Config {
    source_dir: PathBuf,
    output_dir: PathBuf,
    test_ratio: f64,
    val_ratio: f64,
    seed: u64,
    workers: usize,
}

struct CopyJob {
    src: PathBuf,
    dst: PathBuf,
}

// splitmix64: small deterministic PRNG so --seed reproduces the same split without external crates.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.gen_range(i + 1);
        v.swap(i, j);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <source_dir> <output_dir> [seed] [workers]", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  {} './plantvillage dataset/segmented' './plantvillage dataset/dataset-split'",
            args[0]
        );
        eprintln!();
        eprintln!("Splits dataset into train (56%), validation (14%), and test (30%) using a thread pool.");
        std::process::exit(1);
    }

    let default_workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        * 8;

    let cfg = Config {
        source_dir: PathBuf::from(&args[1]),
        output_dir: PathBuf::from(&args[2]),
        test_ratio: 0.3,
        val_ratio: 0.2,
        seed: args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42),
        workers: args
            .get(4)
            .and_then(|s| s.parse().ok())
            .filter(|w| *w > 0)
            .unwrap_or(default_workers),
    };

    if let Err(e) = split_dataset(cfg) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn list_subdirs(dir: &Path) -> io::Result<Vec<String>> {
    let mut dirs: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn list_files(dir: &Path) -> io::Result<Vec<String>> {
    let mut files: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    Ok(files)
}

fn format_duration(d: Duration) -> String {
    let total_sec = d.as_secs();
    format!("{:02}:{:02}", total_sec / 60, total_sec % 60)
}

fn print_progress(done: usize, total: usize, start: Instant) {
    const BAR_WIDTH: usize = 30;

    let pct = if total > 0 {
        done as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let filled = if total > 0 {
        ((BAR_WIDTH as f64) * (done as f64) / (total as f64)) as usize
    } else {
        0
    }
    .min(BAR_WIDTH);
    let bar = "█".repeat(filled) + &"░".repeat(BAR_WIDTH - filled);

    let elapsed = start.elapsed();
    let (speed, eta) = if done > 0 {
        let speed = done as f64 / elapsed.as_secs_f64();
        let remaining = total.saturating_sub(done);
        let eta = if speed > 0.0 {
            Duration::from_secs_f64(remaining as f64 / speed)
        } else {
            Duration::ZERO
        };
        (speed, eta)
    } else {
        (0.0, Duration::ZERO)
    };

    print!(
        "\r\x1b[2K{:3.0}% |{}| {}/{} [{}<{}, {:.0} f/s]",
        pct,
        bar,
        done,
        total,
        format_duration(elapsed),
        format_duration(eta),
        speed
    );
    io::stdout().flush().ok();
}

fn copy_file(src: &Path, dst: &Path) -> io::Result<()> {
    fs::copy(src, dst)?;
    Ok(())
}

fn split_dataset(cfg: Config) -> Result<(), String> {
    let class_names =
        list_subdirs(&cfg.source_dir).map_err(|e| format!("reading source directory: {}", e))?;
    if class_names.is_empty() {
        return Err(format!(
            "no class subdirectories found in {:?}",
            cfg.source_dir
        ));
    }

    println!(
        "Found {} classes in {}",
        class_names.len(),
        cfg.source_dir.display()
    );

    let mut rng = Rng::new(cfg.seed);
    let mut jobs: Vec<CopyJob> = Vec::new();
    let (mut total_train, mut total_val, mut total_test) = (0usize, 0usize, 0usize);

    for class_name in &class_names {
        let class_dir = cfg.source_dir.join(class_name);
        let mut images = list_files(&class_dir)
            .map_err(|e| format!("listing files in {:?}: {}", class_dir, e))?;
        if images.is_empty() {
            continue;
        }

        shuffle(&mut images, &mut rng);

        let test_count = (images.len() as f64 * cfg.test_ratio) as usize;
        let split_idx = images.len() - test_count;
        let (train_val, test_images) = images.split_at(split_idx);

        let val_count = (train_val.len() as f64 * cfg.val_ratio) as usize;
        let train_split_idx = train_val.len() - val_count;
        let (train_images, val_images) = train_val.split_at(train_split_idx);

        let splits: [(&str, &[String]); 3] = [
            ("train", train_images),
            ("validation", val_images),
            ("test", test_images),
        ];

        for (split_name, split_images) in splits {
            let dest_dir = cfg.output_dir.join(split_name).join(class_name);
            fs::create_dir_all(&dest_dir)
                .map_err(|e| format!("creating directory {:?}: {}", dest_dir, e))?;
            for img_name in split_images {
                jobs.push(CopyJob {
                    src: class_dir.join(img_name),
                    dst: dest_dir.join(img_name),
                });
            }
        }

        total_train += train_images.len();
        total_val += val_images.len();
        total_test += test_images.len();
    }

    let total_files = jobs.len();
    println!(
        "Copying {} files with {} worker threads...",
        total_files, cfg.workers
    );

    let start = Instant::now();
    let copied = Arc::new(AtomicUsize::new(0));
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let (tx, rx) = mpsc::channel::<CopyJob>();
    let rx = Arc::new(Mutex::new(rx));

    let mut handles = Vec::with_capacity(cfg.workers);
    for _ in 0..cfg.workers {
        let rx = Arc::clone(&rx);
        let copied = Arc::clone(&copied);
        let errors = Arc::clone(&errors);
        handles.push(thread::spawn(move || loop {
            let job = {
                let lock = rx.lock().unwrap();
                lock.recv()
            };
            match job {
                Ok(job) => {
                    if let Err(e) = copy_file(&job.src, &job.dst) {
                        errors
                            .lock()
                            .unwrap()
                            .push(format!("{} -> {}: {}", job.src.display(), job.dst.display(), e));
                    } else {
                        copied.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => break,
            }
        }));
    }

    let done_flag = Arc::new(AtomicBool::new(false));
    let progress_handle = {
        let copied = Arc::clone(&copied);
        let done_flag = Arc::clone(&done_flag);
        thread::spawn(move || {
            while !done_flag.load(Ordering::Relaxed) {
                print_progress(copied.load(Ordering::Relaxed), total_files, start);
                thread::sleep(Duration::from_millis(200));
            }
            print_progress(copied.load(Ordering::Relaxed), total_files, start);
        })
    };

    for job in jobs {
        tx.send(job).map_err(|e| e.to_string())?;
    }
    drop(tx);

    for h in handles {
        h.join().map_err(|_| "worker thread panicked".to_string())?;
    }
    done_flag.store(true, Ordering::Relaxed);
    progress_handle
        .join()
        .map_err(|_| "progress thread panicked".to_string())?;
    println!();

    let errors = errors.lock().unwrap();
    if !errors.is_empty() {
        for e in errors.iter().take(5) {
            eprintln!("copy error: {}", e);
        }
        if errors.len() > 5 {
            eprintln!("...and {} more errors", errors.len() - 5);
        }
        return Err(format!("{} file(s) failed to copy", errors.len()));
    }

    println!("{}", "─".repeat(60));
    println!("Summary:");
    println!("  Train:      {} images", total_train);
    println!("  Validation: {} images", total_val);
    println!("  Test:       {} images", total_test);
    println!("  Total:      {} images", total_train + total_val + total_test);
    println!("  Output:     {}", cfg.output_dir.display());
    println!("  Workers:    {}", cfg.workers);
    println!("  Elapsed:    {}", format_duration(start.elapsed()));

    Ok(())
}
