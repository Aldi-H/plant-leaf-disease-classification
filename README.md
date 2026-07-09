# Dataset splitting

Splits a PlantVillage-style dataset (one subdirectory per class) into
`train` (56%), `validation` (14%), and `test` (30%) sets, copying files
into `<output_dir>/{train,validation,test}/<class>/`.

Both implementations below do the same split (shuffle with a seed, 70:30
train/test then 80:20 train/validation) and copy files concurrently with a
worker pool, since copying many small files is I/O-latency bound rather
than CPU bound.

## Go (`split_dataset.go`)

Requires Go, no `go.mod`/external dependencies — run or build it by naming
the file directly.

Run directly:

```sh
go run split_dataset.go "<source_dir>" "<output_dir>" [seed] [workers]
```

Or build a binary:

```sh
go build -o split_dataset split_dataset.go
./split_dataset "<source_dir>" "<output_dir>" [seed] [workers]
```

Example:

```sh
go run split_dataset.go "./plantvillage dataset/segmented" "./plantvillage dataset/dataset-split"
```

`seed` defaults to `42`, `workers` defaults to `NumCPU * 8`.

> This directory has no `go.mod`. Always build/run by explicit filename
> (`go run split_dataset.go`), not `go run .` / `go build .` — the latter
> would try to compile every `.go` file in the directory as one package.

## Rust (`split_dataset_parallel.rs`)

Requires `rustc` only — no Cargo project or external crates.

Build:

```sh
rustc -O --edition 2021 split_dataset_parallel.rs -o split_dataset_parallel
```

Run:

```sh
./split_dataset_parallel "<source_dir>" "<output_dir>" [seed] [workers]
```

Example:

```sh
./split_dataset_parallel "./plantvillage dataset/segmented" "./plantvillage dataset/dataset-split"
```

`seed` defaults to `42`, `workers` defaults to `available_parallelism() * 8`.
