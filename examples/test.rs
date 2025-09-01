use tracing::info;

fn main() {
    quantx_core::trace::init_tracing_stdout();
    // let _guard = quantx_core::trace::init_with_persistence();
    let name = "QUANTX-ML";

    info!(target: "Debug", ?name, "Starting up...");
}
