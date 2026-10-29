---
name: sdstl-remote-ssh
description: Use the VS Code integrated terminal to SSH into the SDSTL remote server and run commands under the approved code/results and dataset directories, with strict credential and path handling.
metadata:
  short-description: SSH into SDSTL remote workspace safely
---

# SDSTL Remote SSH

Use this skill when the user asks to inspect, run, edit, or collect results on the remote SDSTL server.

## Connection

- Open a VS Code integrated terminal for remote work.
- Connect with: `ssh root@10.8.135.1 -p 56089`
- Prefer keeping the interactive SSH session open, then run commands from the remote shell.
- Do not store, print, echo, script, or pass the password through command arguments, environment variables, here-strings, files, logs, or automation tools.
- If authentication is needed, let the user type the password into the SSH prompt, or use user-managed SSH key authentication through `ssh-agent` or `%USERPROFILE%\.ssh\config`.
- If the password has appeared in conversation or logs, recommend rotating it and switching to key-based login.

## Allowed Remote Paths

Only work inside these remote trees unless the user gives a new explicit path:

- `/home/user/kpm/kpm`: project code, scripts, checkpoints, logs, and experiment results.
- `/home/user/Datasets/kpm-own`: datasets.

Before changing files, confirm the target path resolves inside one of those roots. Be careful with symlinks and relative paths. A command run from another working directory should `cd` into an allowed root first.

Treat `/home/user/Datasets/kpm-own` as read-only by default. Do not modify, rename, move, or delete dataset files unless the user explicitly asks for that exact dataset operation.

## Command Practice

- For read-only inspection, run commands inside the open SSH terminal when they stay within the allowed roots.
- For edits or experiment runs, state the remote path and intended effect before running the command if the effect is consequential.
- Avoid broad destructive commands. Do not run recursive delete, permission ownership changes, package installs, service changes, process kills, or system configuration edits unless the user explicitly authorizes that specific action.
- Prefer commands that make the working directory explicit after connecting, for example:

```bash
cd /home/user/kpm/kpm
pwd
ls
```

- When command output may include secrets, tokens, private keys, or credentials, summarize the result without reproducing the secret content.
- When editing remote project files, check the relevant context first, preserve unrelated user changes, and keep modifications scoped to the request.

## Useful Orientation

The remote code and result tree mirrors the SDSTL work area. The dataset root uses the updated prefix `/home/user/Datasets/kpm-own`, while older docs or scripts may still mention `/home/user/kpm/kpm/Dataset`; flag or update those paths when that is part of the task.
