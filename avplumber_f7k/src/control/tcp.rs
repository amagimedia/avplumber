//! The line-based control protocol over TCP, the wire format C++ clients
//! speak (`doc/control_protocol.md`): one greeting on connect, one command
//! per line, one status line per command, with a body between `201 OK` and an
//! empty line when there is one.
//!
//! One OS thread per connection over [`exec_line`](super::exec_line); nothing
//! here is asynchronous, because a command is a short call into the core and a
//! client is a person or a script, not a firehose. The listener polls with a
//! short sleep so [`Server::stop`] can end it without a wake-up trick.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::Instance;

/// A running listener. Dropping it stops accepting; connections already open
/// finish their current command and close.
pub struct Server {
    local_addr: SocketAddr,
    stop: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
}

impl Server {
    /// Where the listener actually is, which matters when it was bound to port 0.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.accept_thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Binds `addr` and serves `instance` on it until the returned [`Server`] is
/// stopped or dropped.
pub fn serve(instance: Arc<Instance>, addr: impl ToSocketAddrs) -> std::io::Result<Server> {
    let listener = TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    let local_addr = listener.local_addr()?;
    let stop = Arc::new(AtomicBool::new(false));
    let accept_stop = stop.clone();
    let accept_thread = std::thread::Builder::new()
        .name("avp-control-accept".into())
        .spawn(move || accept_loop(listener, instance, accept_stop))?;
    log::info!("control protocol listening on {local_addr}");
    Ok(Server {
        local_addr,
        stop,
        accept_thread: Some(accept_thread),
    })
}

fn accept_loop(listener: TcpListener, instance: Arc<Instance>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, peer)) => {
                let instance = instance.clone();
                let stop = stop.clone();
                let spawned = std::thread::Builder::new()
                    .name(format!("avp-control-{peer}"))
                    .spawn(move || {
                        if let Err(error) = serve_connection(stream, &instance, &stop) {
                            log::debug!("control connection from {peer} ended: {error}");
                        }
                    });
                if let Err(error) = spawned {
                    log::error!("cannot serve control connection from {peer}: {error}");
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => {
                log::error!("control listener failed: {error}");
                break;
            }
        }
    }
}

fn serve_connection(
    stream: TcpStream,
    instance: &Instance,
    stop: &AtomicBool,
) -> std::io::Result<()> {
    // A blocked read must notice a stop eventually; the timeout only affects
    // how soon, not what the client sees.
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    writer.write_all(b"100 VTR READY\n")?;
    let mut line = String::new();
    loop {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
        let command = line.trim_end_matches(['\r', '\n']);
        if command.trim() == "bye" {
            writer.write_all(b"BYE\n")?;
            return Ok(());
        }
        writer.write_all(reply_for(instance, command).as_bytes())?;
    }
}

/// The status line(s) one command produces, protocol-formatted.
pub fn reply_for(instance: &Instance, command: &str) -> String {
    match super::exec_line(instance, command) {
        Ok(body) if body.is_empty() || body == "ok" => "200 OK\n".into(),
        Ok(body) => format!("201 OK\n{body}\n\n"),
        Err(message) => {
            if let Some(name) = message.strip_prefix("unknown command ") {
                format!("400 Unknown command: {name}\n")
            } else {
                format!("500 ERROR: {message}\n")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

    use super::*;

    fn read_line(reader: &mut BufReader<TcpStream>) -> String {
        let mut line = String::new();
        reader.read_line(&mut line).expect("a reply line");
        line
    }

    #[test]
    fn greets_then_answers_each_command_with_a_status_line() {
        let instance = Arc::new(Instance::new());
        let server = serve(instance, "127.0.0.1:0").expect("bind");
        let stream = TcpStream::connect(server.local_addr()).expect("connect");
        let mut writer = stream.try_clone().unwrap();
        let mut reader = BufReader::new(stream);
        assert_eq!(read_line(&mut reader), "100 VTR READY\n");

        writer.write_all(b"hello\n").unwrap();
        assert_eq!(read_line(&mut reader), "200 OK\n");

        writer.write_all(b"queue.plan_capacity * 1\n").unwrap();
        assert_eq!(read_line(&mut reader), "200 OK\n");

        writer.write_all(b"no.such.command 1 2\n").unwrap();
        assert_eq!(
            read_line(&mut reader),
            "400 Unknown command: no.such.command\n"
        );

        writer.write_all(b"group.status nope\n").unwrap();
        assert!(read_line(&mut reader).starts_with("500 ERROR: "));

        // A body comes back between `201 OK` and an empty line.
        writer
            .write_all(b"node.add {\"type\":\"stub\",\"name\":\"n\",\"group\":\"g\"}\n")
            .unwrap();
        let added = read_line(&mut reader);
        assert!(
            added.starts_with("200") || added.starts_with("500"),
            "node.add answers with a status: {added}"
        );
        writer.write_all(b"bye\n").unwrap();
        assert_eq!(read_line(&mut reader), "BYE\n");
        assert_eq!(
            read_line(&mut reader),
            "",
            "the server closed the connection"
        );
    }

    #[test]
    fn stop_ends_the_listener() {
        let instance = Arc::new(Instance::new());
        let mut server = serve(instance, "127.0.0.1:0").expect("bind");
        let addr = server.local_addr();
        server.stop();
        assert!(
            TcpStream::connect(addr).is_err(),
            "nothing listens after stop"
        );
    }
}
