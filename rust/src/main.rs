//! Train the same agent as `train.py`, in Rust, with no dependencies.
//!
//! This binary answers three questions about the Python version behind it: does
//! the game survive a rewrite, how much of the training time is the interpreter,
//! and can a table trained here feed the existing renderers.
//!
//! Read the file in this order:
//!   `parse_args`   how the flags are read
//!   `train`        the loop that does the work
//!   `evaluate`     the honest measurement
//!   `save_json`    writing a table that Python can load
//!
//! Rust words used in this file:
//!   mod          declares a file as part of the program
//!   fn           a function
//!   struct       a group of named values
//!   &T           a borrow, read only. The value is not copied.
//!   &mut T       a borrow that may change the value
//!   Result<T,E>  either Ok(value) or Err(error). Replaces exceptions.
//!   ?            "if this failed, return the error from this function"
//!   match        compare a value against patterns, like a switch that returns
//!   Some / None  the two halves of an Option
//!
//! Usage:
//!   cargo run --release -- --episodes 20000 --seed 23 --out q_table_rust.json
//!   cargo run --release -- --trace 400 > rust_trace.txt

// Tell the compiler that env.rs and agent.rs belong to this program.
mod agent;
mod env;

// Bring the names used below into scope.
use agent::Agent;
use env::{Flappy, N_ACTIONS, N_DX, N_DY, N_STATES, N_VY, N_Y};
// `std::env` is renamed because this file also has a module called `env`. The
// alias keeps the names apart.
use std::env as std_env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::Instant;

/// The flags, already parsed.
struct Args {
    episodes: usize,
    seed: u64,
    out: String,
    eval_levels: usize,
    trace: usize,
}

/// Read the value that follows a flag, or stop the program with a message.
///
/// `&[String]` is a borrowed slice of the argument list. `i` is the position of
/// the flag itself, so the value sits at i + 1.
fn next_value(argv: &[String], i: usize, flag: &str) -> String {
    // `get` returns Option<&String>. It is the safe way to index: no panic when
    // the position is past the end.
    match argv.get(i + 1) {
        // `.clone()` copies the String. Without it we would return a borrow to
        // data that lives only as long as argv.
        Some(v) => v.clone(),
        None => {
            // `eprintln!` writes to stderr, not stdout. Errors belong there.
            eprintln!("{flag} needs a value");
            // Exit code 2 means "bad usage" by convention.
            std::process::exit(2);
        }
    }
}

/// Read the command line into an Args struct.
fn parse_args() -> Args {
    // `mut` because each flag changes the struct.
    let mut a = Args {
        episodes: 20_000,
        seed: 23,
        // A string literal is `&str`, which cannot change. The struct field is
        // a String, which owns its bytes, so it needs `.to_string()`.
        out: "q_table_rust.json".to_string(),
        eval_levels: 20,
        trace: 0,
    };

    // `std_env::args()` yields every word on the command line, including the
    // program name. `.skip(1)` drops the name. `.collect()` turns the iterator
    // into a Vec<String>.
    let argv: Vec<String> = std_env::args().skip(1).collect();

    // Walk the list two words at a time: one flag, one value.
    let mut i = 0;
    while i < argv.len() {
        // `as_str()` borrows the String as &str, which is what the patterns
        // below compare against.
        match argv[i].as_str() {
            "--episodes" => {
                // `.parse()` converts text to a number. Rust infers the type
                // from the field it is assigned to.
                // `.expect()` panics with this message when the text is not a
                // number. A bad flag is a programmer error, not a user error,
                // so stopping is correct here.
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
            // Two patterns in one arm: `|` means OR.
            "-h" | "--help" => {
                println!("--episodes N  lives to train (20000)");
                println!("--seed N      level seed (23)");
                println!("--out PATH    table to write (q_table_rust.json)");
                println!("--eval-levels N  unseen levels to measure (20)");
                println!("--trace N     print the first N frames of a scripted level");
                // Exit code 0 means "everything is fine".
                std::process::exit(0);
            }
            // The catch all arm. `other` binds whatever did not match.
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

/// The training loop, line for line the one in train.py.
///
/// Returns four values in a tuple: the trained agent, the pipes cleared in every
/// life, the seconds taken, and the number of updates.
fn train(episodes: usize, seed: u64) -> (Agent, Vec<u32>, f64, u64) {
    let mut env = Flappy::new(seed);
    let mut agent = Agent::new(0.20, 0.95, seed);

    // Exploration decays to 2 percent over the first 40 percent of the run,
    // then stays there. Same schedule as the Python trainer.
    //
    // `as f64` converts the count to a float before multiplying. `.max(1.0)`
    // stops a divide by zero when someone asks for zero episodes.
    let decay_lives = ((episodes as f64) * 0.4).max(1.0);

    // A stopwatch.
    let started = Instant::now();

    let mut updates: u64 = 0;
    let mut history: Vec<u32> = Vec::with_capacity(episodes);

    // `1..=episodes` counts from 1 to episodes and includes the last one. The
    // `=` is the difference from `1..episodes`, which stops one short.
    for ep in 1..=episodes {
        // Epsilon goes from about 1.0 down to 0.02 and stays there.
        let epsilon = (1.0 - 0.98 * (ep as f64 / decay_lives)).max(0.02);

        // Start a new level. `obs` is the first state of the life.
        let mut obs = env.reset();

        // One life. `loop` with `break` runs until the episode ends.
        loop {
            let action = agent.act(obs, epsilon);

            // Destructure the tuple that step returns into three names.
            let (next, reward, done) = env.step(action);

            // Learn from the transition, then move to the new state.
            agent.update(obs, action, reward, next, done);
            updates += 1;
            obs = next;

            if done {
                break;
            }
        }

        // Record how many pipes this life cleared.
        history.push(env.passed);
    }

    // `.elapsed()` reads the stopwatch. `.as_secs_f64()` gives seconds as a
    // float rather than whole seconds.
    let secs = started.elapsed().as_secs_f64();
    (agent, history, secs, updates)
}

/// Greedy play on levels the agent never trained on.
///
/// `&Agent` borrows the agent read only. Training is over, nothing should change
/// the table here.
fn evaluate(agent: &Agent, levels: usize) -> (f64, Vec<u32>, usize) {
    let mut pipes = Vec::with_capacity(levels);
    let mut capped = 0;

    // Start at seed 90000, far away from the training seeds.
    for i in 0..levels {
        let mut env = Flappy::new(90_000 + i as u64);
        let mut obs = env.reset();

        loop {
            // greedy, not act. No guessing in a measurement.
            let action = agent.greedy(obs);

            // The underscore throws away the reward. This loop does not learn.
            let (next, _, done) = env.step(action);
            obs = next;
            if done {
                break;
            }
        }

        // A run that ended at the step cap did not crash. Count those
        // separately, because "survived 2400 steps" and "crashed at pipe 70"
        // are different results.
        if env.timed_out {
            capped += 1;
        }
        pipes.push(env.passed);
    }

    // Average. `map` transforms every item, `sum::<f64>()` adds them up, and
    // `.max(1)` avoids a divide by zero when levels is 0.
    let mean = pipes.iter().map(|p| *p as f64).sum::<f64>() / pipes.len().max(1) as f64;
    (mean, pipes, capped)
}

/// The shape of the nested JSON: 6 by 10 by 5 by 10 by 2.
const DIMS: [usize; 5] = [N_DX, N_DY, N_VY, N_Y, N_ACTIONS];

/// Write one level of the nested arrays.
///
/// The table is flat, but the file must be nested, because numpy reads it as a
/// five dimensional array. This function calls itself once per dimension.
///
/// `<W: Write>` makes it generic: it works for any type that can be written to,
/// a file or a memory buffer. `prefix` is the flat index of the current nesting
/// level, and `dim` says which dimension is being written.
///
/// Returns `io::Result<()>`, which is either Ok(()) or an error.
fn write_level<W: Write>(w: &mut W, q: &[f64], prefix: usize, dim: usize) -> io::Result<()> {
    // Open the list.
    write!(w, "[")?;

    for i in 0..DIMS[dim] {
        // A comma between items, but not before the first one.
        if i > 0 {
            write!(w, ",")?;
        }

        // The flat index of this item. At dim 0 this is just i. At the leaf it
        // is the exact position of one score in the table.
        let child = prefix * DIMS[dim] + i;

        // The `?` after each write means: if the write failed, return the error
        // now and stop. It is the short form of a match on Result.
        if dim + 1 == DIMS.len() {
            // Last dimension: write the two numbers.
            // `{:?}` is debug formatting. For a float it prints a decimal point,
            // so 0.0 stays 0.0 and does not turn into 0.
            write!(w, "{:?}", q[child])?;
        } else {
            // Not the last dimension: open a deeper list.
            write_level(w, q, child, dim + 1)?;
        }
    }

    // Close the list. This is the return value, with no semicolon, so its
    // Result is returned to the caller.
    write!(w, "]")
}

/// Same layout agent.py writes and reads: one metadata line, then the table.
fn save_json(path: &str, agent: &Agent) -> io::Result<()> {
    // Create or truncate the file. The `?` propagates a permission error.
    let file = File::create(path)?;

    // BufWriter collects small writes in memory and sends them in big blocks.
    // Without it, every comma would be a separate system call.
    let mut w = BufWriter::new(file);

    // The format string needs a literal brace, written as `{{` and `}}`.
    // `\"` is an escaped quote, because the string is already inside quotes.
    writeln!(w, "{{\"alpha\": {:?}, \"gamma\": {:?}}}", agent.alpha, agent.gamma)?;

    // Then the table itself.
    write_level(&mut w, &agent.q, 0, 0)?;

    // A final newline, then push everything out of the buffer.
    writeln!(w)?;
    w.flush()
}

/// A level with no randomness and a fixed action pattern, so the Python twin
/// can produce the identical frame list.
fn trace(steps: usize) {
    // `scripted` uses the fixed gap ladder instead of the RNG.
    let mut env = Flappy::scripted(1);
    env.reset();

    for t in 0..steps {
        // Flap on every seventh step, coast otherwise. A fixed pattern, so both
        // languages fly the same path.
        let action = if t % 7 == 0 { 1 } else { 0 };
        env.step(action);

        // Four columns: step, height, speed, pipes cleared.
        // {} prints a value, {:.12} prints a float with twelve decimals.
        println!("{} {:.12} {:.12} {}", env.t, env.y, env.vy, env.passed);
    }
}

/// Average a slice of counts. A helper, so main stays short.
///
/// The `<a, I: Iterator<Item = &'a u32>>` is a generic with a lifetime. It says:
/// this works for any iterator that yields borrowed u32 values, and `'a` names
/// how long those borrows last.
fn mean_of<'a, I: Iterator<Item = &'a u32>>(it: I) -> f64 {
    // `.copied()` turns &u32 into u32, `.collect()` gathers them into a Vec.
    let v: Vec<u32> = it.copied().collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<u32>() as f64 / v.len() as f64
    }
}

fn main() {
    // Every Rust program starts here.
    let args = parse_args();

    // The parity mode is not a training run. Do it and leave.
    if args.trace > 0 {
        trace(args.trace);
        return;
    }

    // `let (a, b, c, d) = ...` unpacks the tuple that train returns.
    let (agent, history, secs, updates) = train(args.episodes, args.seed);

    // First 100 lives, last 100 lives, best single life.
    // `.take(100)` keeps the first 100. `.rev()` walks backwards.
    let first = mean_of(history.iter().take(100));
    let last = mean_of(history.iter().rev().take(100));
    let best = history.iter().copied().max().unwrap_or(0);

    // How many of the 6000 numbers are not zero. `.filter` keeps the values
    // that pass the test, `.count()` counts them.
    let visited = agent.q.iter().filter(|v| **v != 0.0).count();

    // One println per line of output. {:.2} is two decimals, {:.1} is one.
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
        // `{:?}` on a Vec prints it like a Python list, with brackets.
        println!(
            "greedy on {} unseen levels: {:.1} pipes ({} hit the step cap) {:?}",
            args.eval_levels, eval_mean, capped, pipes
        );
    }

    // `match` on a Result. Both cases must be handled, so a write error cannot
    // be ignored by accident.
    match save_json(&args.out, &agent) {
        Ok(()) => {
            // Read the file size back, and fall back to 0 if the read fails.
            let kb = std::fs::metadata(&args.out).map(|m| m.len() / 1024).unwrap_or(0);
            println!("wrote {} ({} KB)", args.out, kb);
        }
        Err(e) => {
            eprintln!("could not write {}: {e}", args.out);
            // Exit code 1 means "the program failed".
            std::process::exit(1);
        }
    }
}
