//! Train the same agent as `train.py`, in Rust, with no dependencies.
//!
//! This binary exists to answer three questions about the Python version behind
//! it: does the game survive a rewrite, how much of the training time is the
//! interpreter, and can a table trained here feed the existing renderers.
//!
//! Usage:
//!   cargo run --release -- --episodes 20000 --seed 23 --out q_table_rust.json
//!   cargo run --release -- --trace 400 > rust_trace.txt

mod agent;
mod env;

use agent::Agent;
use env::{Flappy, N_ACTIONS, N_DX, N_DY, N_STATES, N_VY, N_Y};
use std::env as std_env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::Instant;

struct Args {
    episodes: usize,
    seed: u64,
    out: String,
    eval_levels: usize,
    trace: usize,
}

fn next_value(argv: &[String], i: usize, flag: &str) -> String {
    match argv.get(i + 1) {
        Some(v) => v.clone(),
        None => {
            eprintln!("{flag} needs a value");
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        episodes: 20_000,
        seed: 23,
        out: "q_table_rust.json".to_string(),
        eval_levels: 20,
        trace: 0,
    };
    let argv: Vec<String> = std_env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--episodes" => {
                a.episodes = next_value(&argv, i, "--episodes").parse().expect("--episodes");
                i += 2;
            }
            "--seed" => {
                a.seed = next_value(&argv, i, "--seed").parse().expect("--seed");
                i += 2;
            }
            "--out" => {
                a.out = next_value(&argv, i, "--out");
                i += 2;
            }
            "--eval-levels" => {
                a.eval_levels = next_value(&argv, i, "--eval-levels").parse().expect("--eval-levels");
                i += 2;
            }
            "--trace" => {
                a.trace = next_value(&argv, i, "--trace").parse().expect("--trace");
                i += 2;
            }
            "-h" | "--help" => {
                println!("--episodes N  lives to train (20000)");
                println!("--seed N      level seed (23)");
                println!("--out PATH    table to write (q_table_rust.json)");
                println!("--eval-levels N  unseen levels to measure (20)");
                println!("--trace N     print the first N frames of a scripted level");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

/// The training loop, line for line the one in train.py.
fn train(episodes: usize, seed: u64) -> (Agent, Vec<u32>, f64, u64) {
    let mut env = Flappy::new(seed);
    let mut agent = Agent::new(0.20, 0.95, seed);
    // Exploration decays to 2 percent over the first 40 percent of the run,
    // then stays there. Same schedule as the Python trainer.
    let decay_lives = ((episodes as f64) * 0.4).max(1.0);
    let started = Instant::now();
    let mut updates: u64 = 0;
    let mut history: Vec<u32> = Vec::with_capacity(episodes);

    for ep in 1..=episodes {
        let epsilon = (1.0 - 0.98 * (ep as f64 / decay_lives)).max(0.02);
        let mut obs = env.reset();
        loop {
            let action = agent.act(obs, epsilon);
            let (next, reward, done) = env.step(action);
            agent.update(obs, action, reward, next, done);
            updates += 1;
            obs = next;
            if done {
                break;
            }
        }
        history.push(env.passed);
    }

    let secs = started.elapsed().as_secs_f64();
    (agent, history, secs, updates)
}

/// Greedy play on levels the agent never trained on.
fn evaluate(agent: &Agent, levels: usize) -> (f64, Vec<u32>, usize) {
    let mut pipes = Vec::with_capacity(levels);
    let mut capped = 0;
    for i in 0..levels {
        let mut env = Flappy::new(90_000 + i as u64);
        let mut obs = env.reset();
        loop {
            let action = agent.greedy(obs);
            let (next, _, done) = env.step(action);
            obs = next;
            if done {
                break;
            }
        }
        if env.timed_out {
            capped += 1;
        }
        pipes.push(env.passed);
    }
    let mean = pipes.iter().map(|p| *p as f64).sum::<f64>() / pipes.len().max(1) as f64;
    (mean, pipes, capped)
}

const DIMS: [usize; 5] = [N_DX, N_DY, N_VY, N_Y, N_ACTIONS];

/// Nested JSON arrays, the shape numpy expects. Written by hand because the
/// only dependency the format needs is a `[` and a `,`.
fn write_level<W: Write>(w: &mut W, q: &[f64], prefix: usize, dim: usize) -> io::Result<()> {
    write!(w, "[")?;
    for i in 0..DIMS[dim] {
        if i > 0 {
            write!(w, ",")?;
        }
        let child = prefix * DIMS[dim] + i;
        if dim + 1 == DIMS.len() {
            write!(w, "{:?}", q[child])?;
        } else {
            write_level(w, q, child, dim + 1)?;
        }
    }
    write!(w, "]")
}

/// Same layout agent.py writes and reads: one metadata line, then the table.
fn save_json(path: &str, agent: &Agent) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "{{\"alpha\": {:?}, \"gamma\": {:?}}}", agent.alpha, agent.gamma)?;
    write_level(&mut w, &agent.q, 0, 0)?;
    writeln!(w)?;
    w.flush()
}

/// A level with no randomness and a fixed action pattern, so the Python twin
/// can produce the identical frame list.
fn trace(steps: usize) {
    let mut env = Flappy::scripted(1);
    env.reset();
    for t in 0..steps {
        let action = if t % 7 == 0 { 1 } else { 0 };
        env.step(action);
        println!("{} {:.12} {:.12} {}", env.t, env.y, env.vy, env.passed);
    }
}

fn mean_of<'a, I: Iterator<Item = &'a u32>>(it: I) -> f64 {
    let v: Vec<u32> = it.copied().collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<u32>() as f64 / v.len() as f64
    }
}

fn main() {
    let args = parse_args();
    if args.trace > 0 {
        trace(args.trace);
        return;
    }

    let (agent, history, secs, updates) = train(args.episodes, args.seed);

    let first = mean_of(history.iter().take(100));
    let last = mean_of(history.iter().rev().take(100));
    let best = history.iter().copied().max().unwrap_or(0);
    let visited = agent.q.iter().filter(|v| **v != 0.0).count();

    println!(
        "trained {} lives in {:.2}s  ({:.1} M steps/s, {} updates)",
        args.episodes,
        secs,
        updates as f64 / secs / 1e6,
        updates
    );
    println!(
        "cells touched {} of {}   first 100 lives {:.2} pipes   last 100 {:.2}   best {}",
        visited,
        N_STATES,
        first,
        last,
        best
    );

    if args.eval_levels > 0 {
        let (eval_mean, pipes, capped) = evaluate(&agent, args.eval_levels);
        println!(
            "greedy on {} unseen levels: {:.1} pipes ({} hit the step cap) {:?}",
            args.eval_levels, eval_mean, capped, pipes
        );
    }

    match save_json(&args.out, &agent) {
        Ok(()) => {
            let kb = std::fs::metadata(&args.out).map(|m| m.len() / 1024).unwrap_or(0);
            println!("wrote {} ({} KB)", args.out, kb);
        }
        Err(e) => {
            eprintln!("could not write {}: {e}", args.out);
            std::process::exit(1);
        }
    }
}
