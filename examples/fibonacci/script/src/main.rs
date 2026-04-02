use sp1_sdk::{include_elf, utils, ProverClient, SP1Stdin};

const ELF: &[u8] = include_elf!("fibonacci-program");

fn main() {
    utils::setup_logger();

    let n = 100u32;
    let mut stdin = SP1Stdin::new();
    stdin.write(&n);

    let client = ProverClient::from_env();

    let (_, report) = client.execute(ELF, &stdin).run().unwrap();
    println!("executed program with {} cycles", report.total_instruction_count());

    let (pk, vk) = client.setup(ELF);
    let proof = client.prove(&pk, &stdin).core().run().unwrap();
    println!("generated core proof");

    client.verify(&proof, &vk).expect("verification failed");
    println!("successfully generated and verified proof for the program!")
}
