#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Configure jemalloc: immediate page return to OS to minimize RSS under memory pressure.
/// Prevents RSS accumulation across shard boundaries that causes swap thrashing.
#[cfg(not(target_env = "msvc"))]
#[allow(non_upper_case_globals)]
#[export_name = "malloc_conf"]
pub static malloc_conf: &[u8] = b"dirty_decay_ms:0,muzzy_decay_ms:0\0";

use alloy_primitives::B256;
use clap::Parser;
use rsp_client_executor::{io::ClientExecutorInput, CHAIN_ID_ETH_MAINNET};
use std::path::PathBuf;

use sp1_sdk::{include_elf, utils, Prover, ProverClient, SP1Stdin};

#[derive(Parser, Debug)]
struct Args {
    /// Whether or not to generate a proof.
    #[arg(long, default_value_t = false)]
    prove: bool,
}

fn load_input_from_cache(chain_id: u64, block_number: u64) -> ClientExecutorInput {
    let cache_path = PathBuf::from(format!("./input/{}/{}.bin", chain_id, block_number));
    let mut cache_file = std::fs::File::open(cache_path).unwrap();
    let client_input: ClientExecutorInput = bincode::deserialize_from(&mut cache_file).unwrap();

    client_input
}

fn main() {
    utils::setup_logger();
    let args = Args::parse();
    let client_input = load_input_from_cache(CHAIN_ID_ETH_MAINNET, 20526624);

    // Optional cycle limit: CYCLE_LIMIT=600000 → ~1 shard, core-only, no recursion.
    let cycle_limit: Option<u64> = std::env::var("CYCLE_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok());

    let prover = ProverClient::builder().cpu().build();
    let (pk, vk) = prover.setup(include_elf!("rsp-program"));

    let mut stdin = SP1Stdin::new();
    let buffer = bincode::serialize(&client_input).unwrap();
    stdin.write_vec(buffer);

    // Skip full execution when using cycle limit (saves ~2 min).
    if cycle_limit.is_none() {
        let (mut public_values, execution_report) = prover.execute(&pk.elf, &stdin).run().unwrap();
        println!(
            "Finished executing the block in {} cycles",
            execution_report.total_instruction_count()
        );
        let block_hash = public_values.read::<B256>();
        println!("success: block_hash={block_hash}");
    }

    if args.prove {
        println!("Starting proof generation.");

        let proof = if let Some(limit) = cycle_limit {
            println!("Cycle limit set to {limit} (~{} shards, core-only)", limit / 524288 + 1);
            prover.prove(&pk, &stdin)
                .core()
                .cycle_limit(limit)
                .run()
                .expect("Proving should work.")
        } else {
            prover.prove(&pk, &stdin)
                .run()
                .expect("Proving should work.")
        };
        println!("Proof generation finished.");

        if cycle_limit.is_none() {
            prover.verify(&proof, &vk).expect("proof verification should succeed");
        }
    }
}
