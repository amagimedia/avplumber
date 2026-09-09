//! `avplumber` — runs `.avplumber` scripts on the Rust core and serves the
//! control protocol.
//!
//! ```text
//! avplumber [--port PORT] [--bind ADDR] [--log LEVEL] SCRIPT...
//! ```
//!
//! With `--port` the process serves clients until it is killed. Without it, the
//! scripts are run and the process waits for every group member to finish,
//! then exits 0, or 1 when any member failed: a batch transcode.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use avplumber_f7k::{Instance, control};

struct Options {
    port: Option<u16>,
    bind: String,
    log_level: log::LevelFilter,
    scripts: Vec<PathBuf>,
}

const USAGE: &str =
    "usage: avplumber [--port PORT] [--bind ADDR] [--log error|warn|info|debug|trace] SCRIPT...";

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        port: None,
        bind: "0.0.0.0".into(),
        log_level: log::LevelFilter::Info,
        scripts: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                let value = args.next().ok_or("--port needs a value")?;
                options.port = Some(value.parse().map_err(|_| format!("bad port `{value}`"))?);
            }
            "--bind" => options.bind = args.next().ok_or("--bind needs a value")?,
            "--log" => {
                let value = args.next().ok_or("--log needs a value")?;
                options.log_level = value
                    .parse()
                    .map_err(|_| format!("bad log level `{value}`"))?;
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            script => options.scripts.push(PathBuf::from(script)),
        }
    }
    if options.scripts.is_empty() && options.port.is_none() {
        return Err(USAGE.into());
    }
    Ok(options)
}

/// A logger with no dependency: level, target and message on stderr.
struct StderrLogger(log::LevelFilter);

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.0
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{:5}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

/// Every member of every group has reported, or a member failed.
enum Batch {
    Running,
    Finished { failed: bool },
}

fn batch_state(instance: &Instance) -> Batch {
    let mut failed = false;
    for name in instance.group_names() {
        let Some(group) = instance.group(&name) else {
            continue;
        };
        let status = group.status();
        let reported = |member: &str| {
            status.outcomes.iter().any(|outcome| {
                outcome
                    .split(':')
                    .nth(1)
                    .is_some_and(|reported| reported == member)
            })
        };
        if !group.members().iter().all(|member| reported(member)) {
            return Batch::Running;
        }
        failed |= status
            .outcomes
            .iter()
            .any(|outcome| outcome.starts_with("failed:") || outcome.starts_with("panicked:"));
    }
    Batch::Finished { failed }
}

fn main() {
    let options = match parse_args() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let _ = log::set_boxed_logger(Box::new(StderrLogger(options.log_level)));
    log::set_max_level(options.log_level);

    let instance = Arc::new(Instance::new());
    avplumber_nodes::register_media_nodes(&instance);

    for script in &options.scripts {
        if let Err(error) = control::exec_file(&instance, script) {
            log::error!("{}: {error}", script.display());
            std::process::exit(1);
        }
        log::info!("ran {}", script.display());
    }

    match options.port {
        Some(port) => {
            let server = match control::tcp::serve(instance.clone(), (options.bind.as_str(), port))
            {
                Ok(server) => server,
                Err(error) => {
                    log::error!("cannot listen on {}:{port}: {error}", options.bind);
                    std::process::exit(1);
                }
            };
            log::info!("serving on {}", server.local_addr());
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        None => {
            let failed = loop {
                match batch_state(&instance) {
                    Batch::Running => std::thread::sleep(Duration::from_millis(50)),
                    Batch::Finished { failed } => break failed,
                }
            };
            for name in instance.group_names() {
                if let Err(error) = instance.stop_group(&name) {
                    log::warn!("stopping group {name}: {error}");
                }
            }
            // Release the graph — codecs, and with them the hardware device —
            // before leaving. `process::exit` runs libc's exit handlers without
            // running any Rust destructor, and the NVIDIA driver's handler
            // deadlocks joining its own worker thread while a CUDA context is
            // still alive. C++ calls its global destructors here for the same
            // reason.
            drop(instance);
            std::process::exit(if failed { 1 } else { 0 });
        }
    }
}
