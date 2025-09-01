t: example-test

default:
  @just --list

example-test:
  cargo run --example test

test:
  @echo 'Testing!'

build:
  @echo 'Building!'