const BASE = window.__trackio_base || "";

export async function callApi(name, payload = {}) {
  const response = await fetch(`${BASE}/api/${name.replace(/^\//, "")}`, {
    method: "POST",
    credentials: "include",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(sessionStorage.getItem("trackio_oauth_session")
        ? { "x-trackio-oauth-session": sessionStorage.getItem("trackio_oauth_session") }
        : {}),
    },
    body: JSON.stringify(payload),
  });
  const json = await response.json();
  if (!response.ok || json.error) {
    throw new Error(json.error || `Trackio API ${name} failed (${response.status})`);
  }
  return json.data;
}

export function loadRuns(project) {
  return callApi("get_runs_for_project", { project });
}

export function loadRunConfigs(project) {
  return callApi("get_run_configs", { project });
}

export async function loadScalarLogs(project, runs) {
  const entries = [];
  const chunkSize = 64;
  for (let index = 0; index < runs.length; index += chunkSize) {
    const chunk = runs.slice(index, index + chunkSize);
    const result = await callApi("get_logs_batch", {
      project,
      runs: chunk.map((run) => ({ run: run.name, run_id: run.id })),
      max_points: 10000,
      scalar_only: true,
    });
    entries.push(...result);
  }
  return entries;
}
