(() => {
  "use strict";

  const elements = {
    serviceStatus: document.querySelector("#service-status"),
    serviceStatusText: document.querySelector("#service-status-text"),
    workspace: document.querySelector("#workspace"),
    form: document.querySelector("#question-form"),
    question: document.querySelector("#question"),
    questionCount: document.querySelector("#question-count"),
    submitButton: document.querySelector("#submit-button"),
    submitLabel: document.querySelector("#submit-button .button-label"),
    formMessage: document.querySelector("#form-message"),
    loading: document.querySelector("#run-loading"),
    answerPanel: document.querySelector("#answer-panel"),
    answerTitle: document.querySelector("#answer-title"),
    answerSummary: document.querySelector("#answer-summary"),
    verificationBadge: document.querySelector("#verification-badge"),
    verificationLabel: document.querySelector("#verification-label"),
    claimsList: document.querySelector("#claims-list"),
    noClaims: document.querySelector("#no-claims"),
    runMetadata: document.querySelector("#run-metadata"),
    deletionAudit: document.querySelector("#deletion-audit"),
    deletionSummary: document.querySelector("#deletion-summary"),
    deletionList: document.querySelector("#deletion-list"),
    traceList: document.querySelector("#trace-list"),
    traceEmpty: document.querySelector("#trace-empty"),
    stepCount: document.querySelector("#step-count"),
    refreshResults: document.querySelector("#refresh-results"),
    resultsLoading: document.querySelector("#results-loading"),
    resultsEmpty: document.querySelector("#results-empty"),
    resultsError: document.querySelector("#results-error"),
    comparisonContent: document.querySelector("#comparison-content"),
    baselineConfig: document.querySelector("#baseline-config"),
    experimentConfig: document.querySelector("#experiment-config"),
    metricsBody: document.querySelector("#metrics-body"),
  };

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const metricDefinitions = [
    {
      label: "Accuracy",
      aliases: ["accuracy", "answerable_accuracy", "accuracy_rate", "accuracy_pct"],
      format: "percent",
      better: "higher",
    },
    {
      label: "Hallucination rate",
      aliases: ["hallucination_rate", "unanswerable_hallucination_rate", "hallucination_pct"],
      format: "percent",
      better: "lower",
    },
    {
      label: "Verifier catch rate",
      aliases: ["verifier_catch_rate", "citation_verifier_catch_rate", "catch_rate"],
      format: "percent",
      better: "higher",
    },
    {
      label: "Average steps",
      aliases: ["average_steps", "avg_steps", "average_steps_per_question"],
      format: "number",
      better: "neutral",
    },
    {
      label: "Average cost",
      aliases: [
        "average_cost_usd_per_question",
        "average_cost_usd",
        "avg_cost_usd",
        "cost_per_question_usd",
        "average_cost",
      ],
      format: "currency",
      better: "lower",
    },
    {
      label: "Average latency",
      aliases: [
        "average_latency_ms_per_question",
        "average_latency_ms",
        "avg_latency_ms",
        "latency_per_question_ms",
        "average_latency",
      ],
      format: "milliseconds",
      better: "lower",
    },
  ];

  const setHidden = (element, hidden) => {
    element.hidden = hidden;
  };

  const coerceNumber = (value) => {
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }

    if (typeof value === "string") {
      const parsed = Number.parseFloat(value.replace(/[$,%]/g, ""));
      return Number.isFinite(parsed) ? parsed : null;
    }

    if (value && typeof value === "object" && "value" in value) {
      return coerceNumber(value.value);
    }

    return null;
  };

  const requestJson = async (url, options = {}, timeoutMs = 15000) => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {}),
        },
        signal: controller.signal,
      });

      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await response.json() : null;

      if (!response.ok) {
        const detail = payload?.detail || payload?.message || `Request failed (${response.status})`;
        const message = typeof detail === "string" ? detail : JSON.stringify(detail);
        throw new Error(message);
      }

      if (payload === null) {
        throw new Error("The service returned an unexpected response.");
      }

      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("The request timed out before the service responded.");
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  };

  const setServiceStatus = (state, label) => {
    elements.serviceStatus.className = `service-status service-status--${state}`;
    elements.serviceStatusText.textContent = label;
  };

  const checkHealth = async () => {
    setServiceStatus("checking", "Checking");
    try {
      const health = await requestJson("/api/health", {}, 8000);
      if (health?.live_ready === false) {
        setServiceStatus("setup", "Setup needed");
      } else {
        setServiceStatus("online", "Online");
      }
    } catch (_error) {
      setServiceStatus("offline", "Offline");
    }
  };

  const updateCharacterCount = () => {
    elements.questionCount.textContent = `${elements.question.value.length.toLocaleString()} / 2,000`;
  };

  const setRunBusy = (busy) => {
    elements.workspace.setAttribute("aria-busy", String(busy));
    elements.question.disabled = busy;
    elements.submitButton.disabled = busy;
    elements.submitLabel.textContent = busy ? "Researching…" : "Run research";
    setHidden(elements.loading, !busy);
  };

  const sourceBadge = (sourceId) => {
    const badge = document.createElement("span");
    badge.className = "source-badge";
    badge.title = `Evidence source ${sourceId}`;
    badge.setAttribute("aria-label", `Evidence source ${sourceId}`);

    const prefix = document.createElement("span");
    prefix.setAttribute("aria-hidden", "true");
    prefix.textContent = "SRC";

    const value = document.createElement("strong");
    value.textContent = sourceId;

    badge.append(prefix, value);
    return badge;
  };

  const normalizeClaims = (claims) => {
    if (!Array.isArray(claims)) return [];

    return claims.filter(
      (claim) =>
        claim &&
        typeof claim.text === "string" &&
        claim.text.trim() &&
        (typeof claim.source_id === "string" || typeof claim.source_id === "number")
    );
  };

  const renderClaims = (claims) => {
    elements.claimsList.replaceChildren();
    const validClaims = normalizeClaims(claims);

    validClaims.forEach((claim, index) => {
      const item = document.createElement("li");
      const article = document.createElement("article");
      article.className = "claim-card";

      const content = document.createElement("div");
      const label = document.createElement("p");
      label.className = "claim-label";
      label.textContent = `Verified claim ${String(index + 1).padStart(2, "0")}`;

      const text = document.createElement("p");
      text.className = "claim-text";
      text.textContent = claim.text.trim();

      content.append(label, text);
      article.append(content, sourceBadge(String(claim.source_id)));
      item.append(article);
      elements.claimsList.append(item);
    });

    setHidden(elements.claimsList, validClaims.length === 0);
    setHidden(elements.noClaims, validClaims.length > 0);
    elements.answerSummary.textContent = validClaims.length
      ? `${validClaims.length} ${validClaims.length === 1 ? "claim" : "claims"} matched to evidence in this run.`
      : "The verifier did not approve any generated claims.";

    elements.verificationBadge.classList.toggle("verification-badge--empty", validClaims.length === 0);
    elements.verificationLabel.textContent = validClaims.length ? "Verifier passed" : "No verified claims";

    return validClaims.length;
  };

  const addMetadataItem = (term, value) => {
    const wrapper = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value;
    wrapper.append(dt, dd);
    elements.runMetadata.append(wrapper);
  };

  const renderRunMetadata = (data, claimCount) => {
    elements.runMetadata.replaceChildren();
    const latency = coerceNumber(data?.latency_ms);
    const cost = coerceNumber(data?.cost_usd ?? data?.usage?.cost_usd);

    addMetadataItem("Verified", String(claimCount));
    addMetadataItem("Status", String(data?.status || (claimCount ? "complete" : "insufficient information")));
    addMetadataItem("Latency", latency === null ? "Not reported" : formatMilliseconds(latency));
    addMetadataItem("Estimated cost", cost === null ? "Not reported" : formatCurrency(cost));
  };

  const classifyDeletionReason = (deletion) => {
    const reason = typeof deletion === "object" && deletion ? String(deletion.reason || deletion.code || "") : "";
    const normalized = reason.toLowerCase();

    if (normalized.includes("source") && (normalized.includes("missing") || normalized.includes("unknown"))) {
      return "Its source identifier was not present in the step history.";
    }
    if (normalized.includes("failed_source") || (normalized.includes("source") && normalized.includes("fail"))) {
      return "Its cited tool source reported a failure and could not support an answer.";
    }
    if (normalized.includes("support") || normalized.includes("entail") || normalized.includes("ground")) {
      return "The cited source did not support the generated claim.";
    }
    return "The generated claim did not pass citation verification.";
  };

  const renderDeletions = (deletions) => {
    const items = Array.isArray(deletions) ? deletions : [];
    elements.deletionList.replaceChildren();
    setHidden(elements.deletionAudit, items.length === 0);

    if (!items.length) return;

    elements.deletionSummary.textContent = `Verifier audit · ${items.length} ${items.length === 1 ? "claim removed" : "claims removed"}`;
    items.forEach((deletion, index) => {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      const detail = document.createElement("span");
      title.textContent = `Rejected claim ${index + 1}`;
      detail.textContent = classifyDeletionReason(deletion);
      item.append(title, detail);
      elements.deletionList.append(item);
    });
  };

  const humanizeAction = (action) => {
    if (typeof action === "string") return action;
    if (!action || typeof action !== "object") return "Agent step";

    const name = action.tool || action.name || action.type || action.action || "Tool call";
    const detail = action.query || action.input?.query || action.code || action.input?.code || action.input;
    const readableName = String(name).replace(/[_-]+/g, " ");

    if (typeof detail === "string" && detail.trim()) {
      const compact = detail.replace(/\s+/g, " ").trim();
      const truncated = compact.length > 150 ? `${compact.slice(0, 147)}…` : compact;
      return `${readableName}: ${truncated}`;
    }

    if (detail && typeof detail === "object") {
      const serialized = JSON.stringify(detail);
      const truncated = serialized.length > 150 ? `${serialized.slice(0, 147)}…` : serialized;
      return `${readableName}: ${truncated}`;
    }

    return readableName;
  };

  const describeStep = (step) => {
    if (!step || typeof step !== "object") return "Agent step";
    if (step.action === "final") return "Final response and citation verification";
    if (step.action === "model_error") return step.error ? `Model error: ${step.error}` : "Model error";
    if (step.action === "tool_call" && step.tool_name) {
      const description = humanizeAction({ name: step.tool_name, input: step.arguments });
      return step.success === false && step.error ? `${description} — ${step.error}` : description;
    }
    return humanizeAction(step.action);
  };

  const renderTrace = (steps) => {
    const trace = Array.isArray(steps) ? steps : [];
    elements.traceList.replaceChildren();
    setHidden(elements.traceEmpty, trace.length > 0);
    setHidden(elements.traceList, trace.length === 0);
    elements.stepCount.textContent = `${trace.length} ${trace.length === 1 ? "step" : "steps"}`;

    trace.forEach((step, index) => {
      const item = document.createElement("li");
      item.className = "trace-item";

      const marker = document.createElement("span");
      marker.className = "trace-marker";
      marker.setAttribute("aria-hidden", "true");
      marker.textContent = String(step?.step ?? index + 1).padStart(2, "0");

      const content = document.createElement("div");
      content.className = "trace-content";

      const topLine = document.createElement("div");
      topLine.className = "trace-topline";
      const stepLabel = document.createElement("span");
      stepLabel.className = "trace-step-label";
      stepLabel.textContent = `Step ${step?.step ?? index + 1}`;

      const outcome = document.createElement("span");
      const success = step?.success;
      const outcomeState = success === true ? "success" : success === false ? "failure" : "unknown";
      outcome.className = `outcome outcome--${outcomeState}`;
      outcome.textContent = success === true ? "Succeeded" : success === false ? "Failed" : "Recorded";
      topLine.append(stepLabel, outcome);

      const action = document.createElement("p");
      action.className = "trace-action";
      action.textContent = describeStep(step);

      content.append(topLine, action);
      if (step?.source_id !== undefined && step?.source_id !== null && String(step.source_id)) {
        content.append(sourceBadge(String(step.source_id)));
      }

      item.append(marker, content);
      elements.traceList.append(item);
    });
  };

  const renderAnswer = (data) => {
    const claimCount = renderClaims(data?.claims);
    renderRunMetadata(data || {}, claimCount);
    renderDeletions(data?.deletions);
    renderTrace(data?.steps);
    setHidden(elements.answerPanel, false);
    elements.answerTitle.focus({ preventScroll: true });
    elements.answerPanel.scrollIntoView({
      behavior: prefersReducedMotion.matches ? "auto" : "smooth",
      block: "start",
    });
  };

  const runQuestion = async (question) => {
    elements.formMessage.textContent = "";
    setHidden(elements.answerPanel, true);
    setRunBusy(true);

    try {
      const data = await requestJson(
        "/api/ask",
        {
          method: "POST",
          body: JSON.stringify({ question }),
        },
        180000
      );
      renderAnswer(data);
      setServiceStatus("online", "Online");
    } catch (error) {
      elements.formMessage.textContent = error.message || "The research run could not be completed.";
      if (/not configured|api_key/i.test(elements.formMessage.textContent)) {
        setServiceStatus("setup", "Setup needed");
      } else {
        setServiceStatus("offline", "Offline");
      }
      setRunBusy(false);
      elements.question.focus();
    } finally {
      setRunBusy(false);
    }
  };

  const metricContainers = (run) => [
    run,
    run?.metrics,
    run?.summary,
    run?.summary?.metrics,
    run?.aggregate,
    run?.aggregate_metrics,
  ].filter(Boolean);

  const getMetric = (run, aliases) => {
    for (const container of metricContainers(run)) {
      for (const alias of aliases) {
        if (Object.prototype.hasOwnProperty.call(container, alias)) {
          const value = coerceNumber(container[alias]);
          if (value !== null) return value;
        }
      }
    }
    return null;
  };

  const normalizeMetric = (value, format) => {
    if (value === null) return null;
    if (format === "percent" && Math.abs(value) <= 1) return value * 100;
    return value;
  };

  const formatCurrency = (value) => {
    if (value === 0) return "$0.00";
    const digits = Math.abs(value) < 0.01 ? 4 : 2;
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(value);
  };

  const formatMilliseconds = (value) => {
    if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)} s`;
    return `${Math.round(value).toLocaleString()} ms`;
  };

  const formatMetric = (value, format) => {
    if (value === null) return "—";
    if (format === "percent") return `${value.toFixed(1)}%`;
    if (format === "currency") return formatCurrency(value);
    if (format === "milliseconds") return formatMilliseconds(value);
    return value.toFixed(2).replace(/\.00$/, "");
  };

  const formatDelta = (delta, format) => {
    if (delta === null) return "—";
    const sign = delta > 0 ? "+" : delta < 0 ? "−" : "";
    const absolute = Math.abs(delta);

    if (format === "percent") return `${sign}${absolute.toFixed(1)} pp`;
    if (format === "currency") return `${sign}${formatCurrency(absolute)}`;
    if (format === "milliseconds") return `${sign}${formatMilliseconds(absolute)}`;
    return `${sign}${absolute.toFixed(2).replace(/\.00$/, "")}`;
  };

  const deltaClass = (delta, better) => {
    if (delta === null || delta === 0 || better === "neutral") return "delta--neutral";
    const improved = better === "higher" ? delta > 0 : delta < 0;
    return improved ? "delta--positive" : "delta--negative";
  };

  const findConfigValue = (run, aliases) => {
    const containers = [run?.configuration, run?.config, run?.settings, run?.metadata, run].filter(Boolean);
    for (const container of containers) {
      for (const alias of aliases) {
        if (Object.prototype.hasOwnProperty.call(container, alias)) return container[alias];
      }
    }
    return null;
  };

  const describeConfiguration = (run, fallback) => {
    if (!run || typeof run !== "object") return fallback;
    const explicitName = findConfigValue(run, ["label", "name", "configuration_name", "experiment_name"]);
    if (explicitName) return String(explicitName);

    const details = [];
    const maxSteps = findConfigValue(run, ["max_steps", "MAX_STEPS"]);
    const verifier = findConfigValue(run, ["citation_verifier", "verifier_enabled", "use_verifier"]);
    if (maxSteps !== null) details.push(`MAX_STEPS ${maxSteps}`);
    if (verifier !== null) details.push(verifier ? "Verifier on" : "Verifier off");
    return details.length ? details.join(" · ") : fallback;
  };

  const renderMetrics = (baseline, experiment) => {
    elements.metricsBody.replaceChildren();

    metricDefinitions.forEach((definition) => {
      const baselineValue = normalizeMetric(getMetric(baseline, definition.aliases), definition.format);
      const experimentValue = normalizeMetric(getMetric(experiment, definition.aliases), definition.format);
      const delta = baselineValue === null || experimentValue === null ? null : experimentValue - baselineValue;

      const row = document.createElement("tr");
      const labelCell = document.createElement("th");
      const baselineCell = document.createElement("td");
      const experimentCell = document.createElement("td");
      const deltaCell = document.createElement("td");

      labelCell.scope = "row";
      labelCell.textContent = definition.label;
      baselineCell.textContent = formatMetric(baselineValue, definition.format);
      experimentCell.textContent = formatMetric(experimentValue, definition.format);
      deltaCell.textContent = formatDelta(delta, definition.format);
      deltaCell.className = `metric-delta ${deltaClass(delta, definition.better)}`;

      row.append(labelCell, baselineCell, experimentCell, deltaCell);
      elements.metricsBody.append(row);
    });
  };

  const setResultsState = (state, message = "") => {
    setHidden(elements.resultsLoading, state !== "loading");
    setHidden(elements.resultsEmpty, state !== "empty");
    setHidden(elements.resultsError, state !== "error");
    setHidden(elements.comparisonContent, state !== "ready");
    elements.resultsError.textContent = state === "error" ? message : "";
    elements.refreshResults.disabled = state === "loading";
  };

  const loadResults = async () => {
    setResultsState("loading");
    try {
      const data = await requestJson("/api/results", {}, 15000);
      if (!data?.baseline || !data?.experiment) {
        setResultsState("empty");
        return;
      }

      elements.baselineConfig.textContent = describeConfiguration(data.baseline, "Default configuration");
      elements.experimentConfig.textContent = describeConfiguration(data.experiment, "Single-variable change");
      renderMetrics(data.baseline, data.experiment);
      setResultsState("ready");
    } catch (error) {
      setResultsState("error", error.message || "Benchmark results could not be loaded.");
    }
  };

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = elements.question.value.trim();
    if (!question) {
      elements.formMessage.textContent = "Enter a research question before running the agent.";
      elements.question.focus();
      return;
    }
    runQuestion(question);
  });

  elements.question.addEventListener("input", () => {
    updateCharacterCount();
    if (elements.formMessage.textContent) elements.formMessage.textContent = "";
  });

  elements.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      elements.form.requestSubmit();
    }
  });

  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.question.value = button.dataset.question || "";
      updateCharacterCount();
      elements.formMessage.textContent = "";
      elements.question.focus();
    });
  });

  elements.refreshResults.addEventListener("click", loadResults);

  updateCharacterCount();
  checkHealth();
  loadResults();
})();
