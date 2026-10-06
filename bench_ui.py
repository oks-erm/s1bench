"""Local Streamlit dashboard. Run with python start.py."""
from __future__ import annotations
import copy
import json
import queue
import threading
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import benchmark as bench
import local_runtime
from report_export import complete_html_report, METRIC_GUIDE, PROTOCOL_TEXT, MODEL_COLORS
from data_prompt import EXAMPLE, preparation_prompt

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="System 1 benchmark",page_icon="📊",layout="wide")

class RunJob:
    def __init__(self,cfg,cases,selected=None,limit=None,connection_test=False):
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self.finished = False
        self.folder = None
        self.error = None
        self.notified = False
        self.issued = 0
        self.maximum = 0
        self.messages = []
        self.connection_test = connection_test
        cfg,cases = copy.deepcopy(cfg),copy.deepcopy(cases)
        if selected is not None:
            for model in cfg["models"]:
                model["enabled"] = model["name"] in selected
        def work():
            try:
                options = {}
                if any(local_runtime.can_manage(m) for m in cfg["models"] if m.get("enabled")):
                    options["model_session"] = lambda model: local_runtime.model_session(model,self.cancel)
                result = bench.run(cfg,cases,self.cancel,self.events.put,selected,limit,
                                   connection_test=connection_test,**options)
                self.folder,self.error = result["folder"],result["error"]
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.finished = True
        self.thread = threading.Thread(target=work,daemon=True)
        self.thread.start()

def change_config(cfg):
    st.session_state["cfg"] = copy.deepcopy(cfg)
    st.session_state["revision"] += 1
    st.rerun()

def show_data_help(busy):
    st.subheader("Plug your data")
    st.caption("Upload labelled JSONL below, or copy the prompt and attach your source file to an AI.")
    left,right = st.columns(2)
    with left:
        st.markdown("**Every case needs:** id, task, input, question, expected.")
        minimal = {k:EXAMPLE[k] for k in ("id","task","input","question","expected")}
        st.code(json.dumps(minimal,indent=2),language="json")
        st.caption("Shown formatted for readability. The file uses one complete JSON object per line.")
        st.write("Choice gold: option string. Noul gold: boolean. Score gold: numeric level index.")
        st.download_button("Download example JSONL",bench.jsonl_bytes([EXAMPLE]),"example_data.jsonl",
                           mime="application/x-ndjson",disabled=busy)
    with right:
        count = st.number_input("Target cases",100,20000,1400,100,disabled=busy)
        mode = st.selectbox("Source file contains",[
            "Real records; do not invent cases","Reference/policy material; generate synthetic screening cases"
        ],disabled=busy)
        focus = st.text_input("Use-case focus, optional",placeholder="Routing, authorization and escalation",disabled=busy)
        prompt = preparation_prompt(count,mode,focus)
        with st.expander("Copy AI preparation prompt"):
            st.caption("Use the copy icon. Paste into your AI and attach the source file.")
            st.code(prompt,language="text")
        st.download_button("Download preparation prompt",prompt,"prepare_data.txt",disabled=busy)
        st.markdown("**Expected output:** benchmark_data.jsonl plus a short summary in the AI's reply.")
        st.caption("AI-inferred gold stays marked needs_review. Insufficient source records should produce fewer cases, not padding.")

@st.fragment(run_every="1s")
def run_progress():
    job = st.session_state.get("job")
    if not job:
        return
    while True:
        try:
            event = job.events.get_nowait()
        except queue.Empty:
            break
        job.issued,job.maximum = event["issued"],event["maximum"]
        if event.get("message"):
            job.messages.append(event["message"])
            job.messages = job.messages[-100:]
        if event.get("folder"):
            job.folder = event["folder"]
    st.progress(min(job.issued/max(job.maximum,1),1),text=f"Requests issued: {job.issued:,} / maximum {job.maximum:,}")
    if not job.finished:
        if st.button("Stop run",type="primary",disabled=job.cancel.is_set(),key="stop_active_run"):
            job.cancel.set()
        if job.cancel.is_set():
            st.info("Stopping after the current HTTP request. That request can finish and incur a charge; partial results will be saved.")
    if job.error:
        st.error(job.error)
    with st.expander("Run log",expanded=bool(job.error)):
        st.code("\n".join(job.messages[-40:]) or "Preparing run...",language="text")
    if job.finished and not job.notified:
        job.notified = True
        if job.folder:
            st.session_state["report_folder"] = job.folder
        st.rerun()
    if job.finished and job.folder:
        st.success("Saved: "+job.folder)

def protocol_block(report,errors,warnings,stability,robustness):
    with st.expander("Protocol",expanded=False):
        st.write("This measures typed decision components inside an agent, not complete agent success.")
        st.markdown(PROTOCOL_TEXT)
        manifest = report["manifest"]
        st.write({"method":manifest.get("method_version","older run"),
                  "primary_cases":len(report["cases"]),
                  "families":len({c["cluster_id"] for c in report["cases"]}),
                  "interrupted":manifest.get("interrupted"),
                  "request_plan":manifest.get("request_plan"),
                  "started_utc":manifest.get("started_utc")})
        for warning in warnings:
            st.warning(warning)
        for error in errors:
            st.error(error)
        evidence_path = report["folder"]/"evidence.json"
        if evidence_path.exists():
            evidence = bench.parse(evidence_path.read_text("utf-8"))
            st.caption("Saved-file fingerprints match." if not errors else "Evidence checks failed.")
            st.json(evidence)
        st.caption("Unsigned fingerprints detect accidental changes; they do not verify labels or prevent deliberate tampering.")
        if stability:
            st.markdown("**Exact-repeat consistency**")
            display = pd.DataFrame(stability)
            st.dataframe(display,width="stretch")
            st.caption("Agreement is among complete repetition groups; invalid groups count as disagreement. Incomplete groups remain visible.")
        else:
            st.caption("Exact-repeat consistency not measured, or no repeat sample was completed.")
        if robustness:
            st.markdown("**Explicit paired robustness checks**")
            st.dataframe(pd.DataFrame(robustness),width="stretch")
            st.caption("Changed-fact accuracy requires all paired answers correct. Agreement is descriptive; meaningful changes should change the answer.")
        else:
            st.caption("No explicit labelled robustness pairs were supplied.")
        st.caption("Before final selection, freeze policy/rubrics and review critical or ambiguous gold independently. Tune on development data and confirm on a fresh holdout.")

def connection_results_view(report):
    st.subheader("Connection test")
    if report["manifest"].get("interrupted"):
        st.info("Test stopped; completed responses are shown below.")
    status_labels = {"responding":"Responding","invalid_response":"Invalid response",
                     "unavailable":"Failed","skipped":"Skipped","cancelled":"Stopped",
                     "ready":"Not tested"}
    records = {r["model"]:r for r in report["records"]}
    aliases = report["manifest"].get("selected_models",list(report["availability"]))
    models = {m["name"]:m for m in report["config"].get("models",[])}
    rows = []
    for alias in aliases:
        model = models.get(alias,{})
        row = records.get(alias,{})
        status = report["availability"].get(alias,{})
        latency = row.get("latency_s")
        profile = row.get("profile") or {}
        rows.append({"Model":model.get("display_name",alias),
                     "Model ID":row.get("resolved_model") or model.get("model") or "(server default)",
                     "Effort":profile.get("effort","n/a"),
                     "Status":status_labels.get(status.get("status"),"Not tested"),
                     "Latency ms":latency*1000 if bench.finite(latency) else None,
                     "Details":status.get("detail") or row.get("error_detail") or ""})
    st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
    st.caption("Connection status only. Inspect Cases for full responses and errors.")

def result_view(report,cfg):
    st.caption("Loaded report: "+str(report["folder"]))
    if report["manifest"].get("run_kind")=="connection_test":
        connection_results_view(report)
        return
    errors,warnings = bench.audit_report(report)
    analysis_cfg = copy.deepcopy(report["config"])
    # Business target edits are scenarios; captured model settings stay frozen.
    analysis_cfg["business"] = copy.deepcopy(cfg.get("business",bench.default_config()["business"]))
    aliases = [m["name"] for m in analysis_cfg.get("models",[]) if m.get("enabled")]
    aliases = list(dict.fromkeys(aliases+[r["model"] for r in report["records"]]))
    columns = st.columns(2)
    requested = analysis_cfg["business"].get("baseline","gpt")
    baseline_options = aliases if requested in aliases else ["(reference unavailable)"]+aliases
    baseline = columns[0].selectbox("Reference model",baseline_options,
                                   index=baseline_options.index(requested) if requested in baseline_options else 0,
                                   key="report_reference_"+report["folder"].name)
    analysis_cfg["business"]["baseline"] = baseline
    basis = columns[1].selectbox("Cost price basis",["Current GUI prices","Latest saved config","Prices saved with run"])
    override = cfg if basis=="Current GUI prices" else (bench.read_config() if basis=="Latest saved config" else None)
    if errors:
        protocol_block(report,errors,warnings,[],[])
        st.error("Saved evidence checks failed. Comparative results are withheld; inspect Cases or load an unchanged run.")
        return
    import decision_ui
    decision_ui.controls(analysis_cfg,report)
    summaries,records,profiles,quotes,stability,robustness = bench.summarize(
        report["records"],report["cases"],analysis_cfg,report["availability"],override,
        report["manifest"].get("stability_sample_ids"))
    protocol_block(report,errors,warnings,stability,robustness)
    datasets = sorted({r["dataset"] for r in summaries})
    dataset = st.selectbox("Dataset",["All — separate cohorts"]+datasets,key="result_dataset")
    visible = [r for r in summaries if dataset.startswith("All") or r["dataset"]==dataset]
    pending = sum(bench.pending_label(c) for c in report["cases"])
    if pending:
        st.info(f"{pending:,} saved cases have draft/unreviewed labels. Accuracy against those labels is provisional.")
    st.caption("Current business targets are analysis scenarios. Confirm the final choice on a fresh holdout.")
    matrix = []
    for r in visible:
        delta = None if r["delta_vs_baseline"] is None else 100*r["delta_vs_baseline"]
        interval = "Unavailable" if r["delta_ci_low"] is None else f"{100*r['delta_ci_low']:.1f} to {100*r['delta_ci_high']:.1f}"
        matrix.append({
            "Dataset":r["dataset"],"Use case":r["task"],
            "Model":f"{r['display_name']} · {r['requested_model']} · effort={r['effort']}",
            "Status":r["status"],"Success %":100*r["success_rate"] if r["success_rate"] is not None else None,
            "Evaluation scope":r['evaluation_scope'],"Deployment eligibility":r['deployment_eligibility'],
            "Missing decision targets":r['outstanding_targets'],
            "Blocked business decisions":r['blocked_business_decisions'],
            "Non-review-label coverage %":100*r['label_based_coverage'] if r['label_based_coverage'] is not None else None,
            "Threshold-qualified coverage %":100*r['threshold_coverage'] if r['threshold_coverage'] is not None else None,
            "Business-policy-eligible coverage %":100*r['business_eligible_coverage'] if r['business_eligible_coverage'] is not None else None,
            "Gap vs reference pp":delta,"Gap 95% interval pp":interval,
            "p95 ms":r["p95_ms"],"API / 1k":r["api_per_1k"],"Currency":r["currency"],
            "Cost coverage %":100*r["cost_coverage"] if r["cost_coverage"] is not None else None,
            "Critical failures":f"{r['critical_failures']}/{r['critical_cases']}",
            "Unsafe decisions":f"{r['unsafe_decisions']}/{r['risk_exposures']}",
            "Cases / families":f"{r['attempted']}/{r['families']}",
            "Scenario net hours saved":r['workflow_scenario']['net_hours_saved'],
            "Scenario review hours":r['workflow_scenario']['review_hours'],
            "Scenario total / 1k":r['workflow_scenario']['total_per_1k'],
            "Scenario currency":r['workflow_scenario']['currency'],
            "Next step":r["recommendation"],
        })
    st.subheader("Business decision matrix")
    st.dataframe(pd.DataFrame(matrix),width="stretch",hide_index=True)
    cohorts = sorted({(r["dataset"],r["task"]) for r in visible})
    if not cohorts:
        return
    cohort = st.selectbox("Inspect one use case",cohorts,format_func=lambda c:c[0]+" / "+c[1])
    scoped = [r for r in visible if (r["dataset"],r["task"])==cohort]
    visible_keys = {(r["model"],r["dataset"],r["task"]) for r in visible}
    confusion = [table for table in bench.confusion_tables(records)
                 if (table["model"],table["dataset"],table["task"]) in visible_keys]
    df = pd.DataFrame(scoped)
    complete = df[df["status"]=="complete"].copy()
    tabs = st.tabs(["Quality","Latency","Cost and forecast","Fallback scenario","Profiles and evidence","Confusion matrices"])
    with tabs[0]:
        st.dataframe(df[["model","planned","attempted","families","success_rate","success_ci_low","success_ci_high",
                         "ci_method","macro_f1","brier","mae","valid_rate","api_errors","invalid_answers",
                         "none_precision","none_recall","non_none_on_none_rate","unsafe_decisions",
                         "risk_exposures","unsafe_upper95","risk_direction","auto_coverage","auto_accuracy"]].rename(columns={"auto_coverage":"Label-based automation coverage","auto_accuracy":"Label-based accepted accuracy"}),width="stretch")
        if not complete.empty:
            fig = px.bar(complete,x="model",y="success_rate",hover_data=["requested_model","effort","attempted","families"],
                         color="model",color_discrete_map=MODEL_COLORS,
                         color_discrete_sequence=px.colors.qualitative.Safe,template="plotly_white",
                         labels={"model":"Model","success_rate":"Task success"},
                         title="Task success on complete matched cohorts")
            fig.update_yaxes(range=[0,1],tickformat=".0%")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig,width="stretch",theme=None)
        st.caption("Critical failures are test failures, not measured production harm. Unsafe errors follow the declared risk direction and safe labels.")
        pairs = [r for r in robustness if r["task"]==cohort[1]]
        if pairs:
            st.dataframe(pd.DataFrame(pairs),width="stretch")
    with tabs[1]:
        st.dataframe(df[["model","p50_ms","p95_ms","attempt_p95_ms","timed_completed","api_errors"]],width="stretch")
        if not complete.empty:
            fig = px.bar(complete,x="model",y=["p50_ms","p95_ms"],barmode="group",
                         color_discrete_map={"p50_ms":"#0072B2","p95_ms":"#E69F00"},template="plotly_white",
                         labels={"model":"Model","value":"Latency (ms)","variable":"Percentile"},
                         title="Client latency, milliseconds")
            st.plotly_chart(fig,width="stretch",theme=None)
        st.caption("p95 needs enough observations. These are sequential client timings, not concurrent load or throughput tests.")
    with tabs[2]:
        st.subheader("Estimated API cost for this run")
        totals = [{"Model":row["model"],"Estimated API cost":row["estimated_api_cost"],
                   "Currency":row["currency"],"Requests priced":f"{row['priced_requests']} / {row['requests']}",
                   "Input / 1M tokens":row["input_per_million"],"Output / 1M tokens":row["output_per_million"],
                   "Price basis":row["rate_basis"]} for row in bench.cost_totals(records,quotes)]
        st.dataframe(pd.DataFrame(totals),width="stretch",hide_index=True,
                     column_config={"Estimated API cost":st.column_config.NumberColumn(format="%.6f")})
        st.caption("Includes primary cases, warm-ups and repeats in this saved run. These are rate-based estimates, not provider invoices. Blank current prices fall back to saved rates; the applied rate basis is shown for each model.")
        fields = ["model","api_per_1k","cost_per_correct","cost_coverage","active_hosting_per_1k","active_total_per_1k",
                  "monthly_api","always_on_30day_hosting","currency","rate_basis"]
        st.dataframe(df[fields],width="stretch")
        savings = [r for r in scoped if r.get("monthly_api_savings_vs_baseline") is not None]
        if savings:
            st.dataframe(pd.DataFrame([{"model":r["model"],"monthly_API_savings":r["monthly_api_savings_vs_baseline"],
                                       "currency":r["currency"]} for r in savings]),width="stretch")
        known = sum(r["_api_cost"] for r in records if r["_api_cost"] is not None)
        unknown = sum(r["_api_cost"] is None for r in records)
        currencies = {quotes[r["model"]].get("pricing_currency","USD") for r in records}
        if len(currencies)==1:
            st.write({"known_API_spend_all_phases":known,"currency":next(iter(currencies)),
                      "requests_with_unknown_cost":unknown})
        else:
            st.caption("Mixed currencies: spend is not summed.")
        st.caption("Hosting / 1k is a serial active-time scenario. Always-on hosting assumes 720 hours, separate from API fees.")
    with tabs[3]:
        choices = sorted({r["decision"] for r in records if bench.primary_row(r) and r["dataset"]==cohort[0]
                          and r["task"]==cohort[1] and r["type"]=="choice" and r["valid"]})
        labels = st.multiselect("Also fall back on these explicit choice labels",choices,
                               default=[x for x in choices if x in {"NONE","CLARIFY","ESCALATE"}])
        use_acceptance=st.checkbox("Also fall back when the selected acceptance policy defers",value=False)
        if use_acceptance:
            replay=[]
            for r in scoped:
                replay += [x for x in bench.fallback_replay(records,summaries,*cohort,baseline,set(labels),r["acceptance_policy"]) if x["model"]==r["model"]]
        else:
            replay = bench.fallback_replay(records,summaries,*cohort,baseline,set(labels))
        st.dataframe(pd.DataFrame(replay),width="stretch")
        st.caption("Offline replay: invalid/API-failed decisions always fall back. Only complete, matched, same-currency cohorts qualify. Latencies add serially. No inference calls are made.")
    with tabs[4]:
        st.json(profiles)
        for alias,quote in quotes.items():
            st.write(alias,{"rates_per_million":quote.get("pricing_per_million"),
                            "basis":quote["rate_basis"],"lookup":quote.get("pricing_lookup"),
                            "hosting_cost_per_hour":quote.get("hosting_cost_per_hour")})
        st.json(report["manifest"])
        st.caption("Checkpoint/encoder fields are configured provenance, not independently verified server identity.")
    with tabs[5]:
        st.caption("Rows are expected labels; columns are predictions. Counts include attempted primary cases only. Warm-ups and repeats are excluded. Missing requests are not counted; check coverage in Quality.")
        matrices = [table for table in confusion if (table["dataset"],table["task"])==cohort]
        if not matrices:
            st.info("Confusion matrices apply to choice and yes/no tasks with recorded primary responses. For score tasks, see Quality for numeric errors.")
        for table in matrices:
            frame = pd.DataFrame.from_dict(table["counts"],orient="index").fillna(0).astype(int)
            labels = sorted(set(frame.index) | (set(frame.columns)-{"(invalid/API error)"}))
            predicted = labels + (["(invalid/API error)"] if "(invalid/API error)" in frame.columns else [])
            frame = frame.reindex(index=labels,columns=predicted,fill_value=0)
            fig = px.imshow(frame,text_auto=True,aspect="auto",color_continuous_scale="Blues",zmin=0,
                            labels={"x":"Predicted label","y":"Expected label","color":"Cases"},
                            title=table["model"]+" · "+cohort[0]+" / "+cohort[1],template="plotly_white")
            st.plotly_chart(fig,width="stretch",theme=None)
    decision_ui.render(scoped,records,report["cases"],analysis_cfg,report)
    st.download_button("Download business CSV",bench.csv_bytes(visible),"business_report.csv",mime="text/csv")
    st.caption("The complete HTML includes every dataset, use case, result tab and saved case, regardless of the filters above. Recipients can explore it offline; it contains case inputs and responses, but no API-key settings.")
    st.download_button("Download complete HTML report",
                       complete_html_report(report,analysis_cfg=analysis_cfg,current_prices=override,
                           analysis=(summaries,records,profiles,quotes,stability,robustness),price_basis=basis),
                       "benchmark_complete_report.html",mime="text/html")
    st.session_state["analysis_records"] = records

def case_view(report):
    rows = report["records"]
    if not rows:
        st.info("No completed requests in this run.")
        return
    model = st.selectbox("Model",["All"]+sorted({r["model"] for r in rows}),key="case_model")
    task = st.selectbox("Use case",["All"]+sorted({r["task"] for r in rows}),key="case_task")
    mode = st.selectbox("Cases",["Failures","All primary cases","Warmup / connection checks","Repeated sample"],
                        index=2 if report["manifest"].get("run_kind")=="connection_test" else 0,
                        key="case_mode_"+report["folder"].name)
    selected = [r for r in rows if (model=="All" or r["model"]==model) and (task=="All" or r["task"]==task)]
    if mode=="Failures":
        selected = [r for r in selected if bench.primary_row(r) and not r["correct"]]
    elif mode=="All primary cases":
        selected = [r for r in selected if bench.primary_row(r)]
    elif mode=="Repeated sample":
        selected = [r for r in selected if r.get("phase")=="repeat"]
    else:
        selected = [r for r in selected if r.get("phase") in {"warmup","connection_check"}]
    if not selected:
        st.info("No cases match these filters.")
        return
    st.dataframe(pd.DataFrame([{k:r.get(k) for k in ("model","id","task","phase","expected","value","correct","valid",
                                                   "api_error","error_detail","latency_s","input_tokens","output_tokens")}
                              for r in selected]),width="stretch")
    index = st.selectbox("Inspect a request",range(len(selected)),
                         format_func=lambda i:f"{selected[i]['model']} / {selected[i]['id']} / {selected[i].get('phase','primary')}")
    row = selected[index]
    case = next((c for c in report["cases"] if c["id"]==row["id"]),None)
    if case:
        st.markdown("**Original case and frozen question**")
        st.json(case)
    st.markdown("**Normalized prediction**")
    st.json({k:v for k,v in row.items() if k!="raw"})
    with st.expander("Full raw response and API error",expanded=bool(row.get("api_error"))):
        raw = row.get("raw")
        if isinstance(raw,(dict,list)):
            st.json(raw)
        else:
            st.code(str(raw),language="text")
        if row.get("error_detail"):
            st.code(row["error_detail"],language="text")

def optional_number(label,value,key,disabled=False):
    text = st.text_input(label,"" if value is None else str(value),key=key,disabled=disabled)
    if not text.strip():
        return None
    try:
        number = float(text)
        if not bench.finite(number) or number < 0:
            raise ValueError()
        return number
    except ValueError:
        st.error(label+": enter a nonnegative number or leave blank.")
        st.session_state.setdefault("input_errors",[]).append(label)
        return value

def model_editor(model,revision,busy):
    alias = model["name"]
    key = lambda field:f"m_{revision}_{alias}_{field}"
    with st.expander(f'{model.get("display_name",alias)} · {model.get("model") or "server default"}',expanded=model.get("enabled",False)):
        a,b = st.columns(2)
        model["display_name"] = a.text_input("Display name",model.get("display_name",alias),key=key("display"),disabled=busy)
        adapters = ["systemone","openai","ollama"]
        api = model.get("api","systemone")
        model["api"] = b.selectbox("API adapter",adapters,index=adapters.index(api) if api in adapters else 0,
                                  key=key("api"),disabled=busy)
        model["endpoint"] = st.text_input("Full inference URL",model.get("endpoint",""),key=key("endpoint"),disabled=busy)
        model["model"] = st.text_input("Exact model ID (blank only for SystemOne server default)",model.get("model",""),
                                     key=key("model"),disabled=busy)
        left,right = st.columns(2)
        model["api_key"] = left.text_input("API key",model.get("api_key",""),type="password",key=key("key"),disabled=busy)
        model["api_key_env"] = right.text_input("Or environment variable containing the key",model.get("api_key_env",""),
                                             key=key("env"),disabled=busy)
        deploys = ["local","hosted","remote"]
        deployment = model.get("deployment","hosted")
        model["deployment"] = st.selectbox("Deployment / billing",deploys,index=deploys.index(deployment) if deployment in deploys else 1,
                                           key=key("deployment"),disabled=busy)
        eligibility=["unknown","approved","not approved"]
        model["deployment_eligibility"]=st.selectbox("Organisation-declared deployment eligibility",eligibility,
            index=eligibility.index(model.get("deployment_eligibility","unknown")),key=key("eligibility"),disabled=busy)
        st.caption("Supplied by your organisation; model quality and hosted/local location do not grant approval.")
        params_text = st.text_area("Additional API parameters (JSON)",json.dumps(model.get("params",{}),indent=2),
                                  key=key("params"),disabled=busy,height=100)
        try:
            params = bench.parse(params_text)
            if not isinstance(params,dict):
                raise ValueError("Parameters must be an object.")
            model["params"] = params
        except ValueError as exc:
            st.error(str(exc))
            st.session_state["input_errors"].append(alias+": parameters")
        if api == "openai":
            st.caption('GPT Responses effort: {"reasoning":{"effort":"low"}} if your model supports it. Chat endpoint: {"reasoning_effort":"low"}. No unsupported options are added automatically.')
        rates = model.get("pricing_per_million",{})
        if bench.local_model(model):
            st.info("Local API fee: $0. Add hosting cost below to compare infrastructure.")
        else:
            x,y = st.columns(2)
            with x:
                rate_in = optional_number("Input price per million tokens",rates.get("input"),key("rate_in"),busy)
            with y:
                rate_out = optional_number("Output price per million tokens",rates.get("output"),key("rate_out"),busy)
            if rate_in is None and rate_out is None:
                model.pop("pricing_per_million",None)
            elif rate_in is not None and rate_out is not None:
                if rate_in != rates.get("input") or rate_out != rates.get("output"):
                    model.pop("pricing_lookup",None)
                    model["pricing_per_million"] = {"input":rate_in,"output":rate_out,"currency":"USD",
                                                    "source":"manual","retrieved_at":bench.utc()}
            else:
                st.warning("Supply both prices, including 0 if appropriate.")
                st.session_state["input_errors"].append(alias+": incomplete prices")
            if model.get("pricing_per_million"):
                st.caption("Pricing source: "+str(model["pricing_per_million"].get("source","manual")))
        model["hosting_cost_per_hour"] = optional_number("Hosting USD/hour (optional scenario)",model.get("hosting_cost_per_hour"),
                                                   key("hosting"),busy)
        with st.expander("Serving provenance and registry lookup"):
            for field,label in [("checkpoint_id","Checkpoint ID"),("checkpoint_revision","Checkpoint revision"),
                                ("encoder_id","Encoder ID"),("encoder_revision","Encoder revision"),
                                ("serving_notes","Context limits, quantization, cache and server settings")]:
                model[field] = st.text_input(label,model.get(field,""),key=key(field),disabled=busy)
            model["pricing_catalog_key"] = st.text_input("Exact LiteLLM price registry key, optional",
                                                        model.get("pricing_catalog_key",""),key=key("catalog"),disabled=busy)
        if st.button("Discover served model IDs (metadata only)",key=key("discover"),disabled=busy):
            try:
                st.session_state[key("discovered")] = bench.discover_models(model)
            except Exception as exc:
                st.error(str(exc))
        discovered = st.session_state.get(key("discovered"),[])
        if discovered:
            chosen = st.selectbox("Available model IDs",discovered,key=key("chosen"),disabled=busy)
            if st.button("Use this ID",key=key("apply"),disabled=busy):
                model["model"] = chosen
                change_config(st.session_state["cfg"])
        problem = bench.model_problem(model)
        if problem:
            st.warning(problem)
        elif model.get("enabled") and not bench.local_model(model) and model.get("api_key_env") and not bench.model_key(model):
            st.warning("No API key found. This model will be skipped; other enabled models can run.")

def settings_view(cfg,revision,busy):
    with st.expander("Protocol and business targets"):
        p = cfg.setdefault("protocol",bench.default_config()["protocol"].copy())
        a,b,c = st.columns(3)
        p["warmup_calls"] = a.number_input("Warm-up calls/model",0,10,int(p.get("warmup_calls",2)),key=f"p_warm_{revision}",disabled=busy)
        p["batches"] = b.number_input("Balanced batches",1,50,int(p.get("batches",10)),key=f"p_batches_{revision}",disabled=busy)
        p["repeat_cases"] = c.number_input("Cases in repeat subset",0,5000,int(p.get("repeat_cases",100)),key=f"p_repeat_{revision}",disabled=busy)
        p["repetitions"] = a.number_input("Total observations per repeated case",1,10,int(p.get("repetitions",3)),key=f"p_reps_{revision}",disabled=busy)
        p["bootstrap_samples"] = b.number_input("Family bootstrap samples",100,10000,int(p.get("bootstrap_samples",1000)),100,key=f"p_boot_{revision}",disabled=busy)
        p["max_requests"] = c.number_input("Maximum requests per run",1,1000000,int(p.get("max_requests",10000)),key=f"p_cap_{revision}",disabled=busy)
        p["score_repeat_tolerance"] = a.number_input("Score repeat tolerance",0.0,10.0,float(p.get("score_repeat_tolerance",.25)),.05,key=f"p_tol_{revision}",disabled=busy)
        cfg["timeout_s"] = b.number_input("Request timeout, seconds",1.0,600.0,float(cfg.get("timeout_s",45)),key=f"p_timeout_{revision}",disabled=busy)
        cfg["seed"] = c.number_input("Sampling seed",0,2147483647,int(cfg.get("seed",42)),key=f"p_seed_{revision}",disabled=busy)
        biz = cfg.setdefault("business",bench.default_config()["business"].copy())
        aliases = [m["name"] for m in cfg["models"]]
        if aliases:
            baseline = biz.get("baseline","gpt")
            biz["baseline"] = st.selectbox("Reference model alias",aliases,index=aliases.index(baseline) if baseline in aliases else 0,
                                           key=f"b_base_{revision}",disabled=busy)
        biz["max_quality_drop_pp"] = a.number_input("Maximum quality drop vs reference, percentage points",0.0,100.0,float(biz.get("max_quality_drop_pp",2)),key=f"b_drop_{revision}",disabled=busy)
        biz["min_cases"] = b.number_input("Minimum cases per use case",1,100000,int(biz.get("min_cases",200)),key=f"b_cases_{revision}",disabled=busy)
        biz["min_families"] = c.number_input("Minimum independent families",1,100000,int(biz.get("min_families",100)),key=f"b_families_{revision}",disabled=busy)
        biz["max_p95_ms"] = optional_number("Maximum p95 latency, ms",biz.get("max_p95_ms"),f"b_latency_{revision}",busy)
        biz["max_risk_percent"] = optional_number("Maximum harmful error rate, %",biz.get("max_risk_percent"),f"b_risk_{revision}",busy)
        biz["min_consistency_percent"] = optional_number("Minimum repeat agreement, %",biz.get("min_consistency_percent"),f"b_consistency_{revision}",busy)
        biz["monthly_volume"] = st.number_input("Monthly production decisions (scenario)",0,1000000000,int(biz.get("monthly_volume",100000)),key=f"b_volume_{revision}",disabled=busy)
        st.caption("Freeze targets before the holdout run. Ten batches partition this dataset; they do not create ten independent datasets.")

bench.initialize()
if "cfg" not in st.session_state:
    try:
        st.session_state["cfg"] = bench.read_config(BASE/"config.json")
    except Exception as exc:
        st.session_state["cfg"] = bench.default_config()
        st.session_state["config_load_error"] = str(exc)
    st.session_state["revision"] = 0
cfg = st.session_state["cfg"]
revision = st.session_state["revision"]
job = st.session_state.get("job")
busy = bool(job and not job.finished)
st.session_state["input_errors"] = []

st.title("System 1 benchmark")
st.caption("Compare model quality, speed, cost and consistency on your data.")
# Keep one stable slot above navigation: adding progress must not reset the active tab.
with st.container():
    if st.session_state.get("config_load_error"):
        st.warning("Saved config could not be loaded: "+st.session_state["config_load_error"])
    if busy:
        st.warning("A run is active. Use Stop below to prevent further requests.")
    run_progress()
models_tab,data_tab,run_tab,results_tab,cases_tab,guide_tab = st.tabs(
    ["Models & config","Plug your data","Run","Results","Cases","Metric guide"])

with models_tab:
    st.subheader("Model settings")
    st.caption("Configure endpoints, model IDs and keys here. Choose which models to run on the Run tab.")
    upload_cfg = st.file_uploader("Load configuration",type=["json"],disabled=busy,key="upload_cfg")
    if upload_cfg:
        digest = bench.fingerprint(upload_cfg.getvalue().decode("utf-8-sig"))
        if digest != st.session_state.get("cfg_upload_digest"):
            try:
                imported = bench.parse(upload_cfg.getvalue().decode("utf-8-sig"))
                bench.validate_config(imported)
                st.session_state["cfg_upload_digest"] = digest
                change_config(imported)
            except Exception as exc:
                st.error(str(exc))
    for model in cfg["models"]:
        model_editor(model,revision,busy)
    settings_view(cfg,revision,busy)
    if st.button("Look up recent API prices",disabled=busy):
        with st.spinner("Checking the price registry; manual prices are preserved"):
            notes = bench.refresh_prices(cfg,force=False)
        st.session_state["price_notes"] = notes
        change_config(cfg)
    for note in st.session_state.get("price_notes",[]):
        st.info(str(note))
    remember = st.checkbox("Include entered keys when saving locally",value=False,disabled=busy)
    to_save = copy.deepcopy(cfg) if remember else bench.scrub_config(cfg)
    a,b = st.columns(2)
    if a.button("Save config.json",disabled=busy or bool(st.session_state["input_errors"])):
        try:
            bench.validate_config(cfg)
            bench.write_json(BASE/"config.json",to_save)
            st.success("Saved config.json. API keys are "+("included." if remember else "kept only in this session; use environment variables for later launches."))
        except Exception as exc:
            st.error(str(exc))
    b.download_button("Download configuration",json.dumps(to_save,indent=2),"config.json",
                       mime="application/json",disabled=busy)

cases = []
with data_tab:
    show_data_help(busy)
    source = st.radio("Dataset source",["Starter screening suite","Upload your files","Local path or dataset manifest"],
                      horizontal=True,disabled=busy,key="data_source")
    try:
        if source=="Starter screening suite":
            cases = bench.load_data(BASE/"data"/"starter.jsonl",cfg)
            st.info("1,400 synthetic fixtures across 12 tasks. Their related variants share families. Use reviewed domain data for business decisions.")
        elif source=="Upload your files":
            role = st.selectbox("Role of these datasets",["external","production_holdout","development","synthetic"],
                                disabled=busy,key="upload_role")
            uploads = st.file_uploader("Labelled JSONL files",type=["jsonl"],accept_multiple_files=True,
                                       disabled=busy,key="upload_data")
            for item in uploads:
                cases.extend(bench.load_jsonl(item.getvalue(),item.name,cfg,role=role))
        else:
            path = st.text_input("JSONL or manifest .json path",str(BASE/"data"/"starter.jsonl"),disabled=busy,key="data_path")
            if path:
                cases = bench.load_data(Path(path).expanduser(),cfg)
        if cases:
            warnings = bench.validate_data(cases,cfg)
            a,b,c = st.columns(3)
            a.metric("Cases",len(cases))
            b.metric("Tasks",len({c["task"] for c in cases}))
            c.metric("Independent families",len({c["cluster_id"] for c in cases}))
            for note in warnings:
                st.warning(note)
            st.dataframe(pd.DataFrame([{"id":c["id"],"dataset":c.get("dataset"),"task":c["task"],
                                         "type":c["question"]["type"],"gold":str(c["expected"]),
                                         "family":c["cluster_id"],"review":c.get("review_status","unspecified"),
                                         "role":c.get("dataset_role")} for c in cases]),width="stretch")
            inspect = st.selectbox("View a data case",range(len(cases)),format_func=lambda i:cases[i]["id"],key="data_case")
            st.json(cases[inspect])
            st.download_button("Download current cases",bench.jsonl_bytes(cases),"benchmark_data.jsonl",
                               mime="application/x-ndjson",disabled=busy)
        else:
            st.info("Choose a labelled data file to enable the benchmark.")
    except Exception as exc:
        st.error("Dataset validation: "+str(exc))
        cases = []

with run_tab:
    st.subheader("Choose models")
    st.caption("Select models for one comparison report. Configure their connections in Models & config.")
    selected = []
    for model in cfg["models"]:
        label = model.get("display_name",model["name"])
        profile = bench.captured_profile(model)
        details = ["Local" if bench.local_model(model) else "API", profile["requested_model"]]
        if profile["effort"] not in ("n/a", "unspecified"):
            details.append("Reasoning: "+str(profile["effort"]))
        model["enabled"] = st.checkbox(label,value=bool(model.get("enabled")),
                                       key=f"run_model_{revision}_{model['name']}",
                                       help=" · ".join(details),disabled=busy)
        if model["enabled"]:
            selected.append(model["name"])
    if not selected:
        st.info("Tick the models you want to run.")
    if any(local_runtime.can_manage(m) for m in cfg["models"] if m.get("enabled")):
        st.info("Installed local models start and stop automatically, one at a time. All selected models share one report. Stop any separately launched model server first.")
    if not cases:
        st.info("Choose a dataset in Plug your data before running a benchmark or connection test.")
    plan = None
    run_cfg = copy.deepcopy(cfg)
    try:
        bench.validate_config(run_cfg)
        if selected and cases:
            plan = bench.request_plan(cases,run_cfg,selected)
            a,b,c = st.columns(3)
            a.metric("Dataset cases",len(cases))
            b.metric("Selected models",len(selected))
            c.metric("Maximum benchmark requests",plan["maximum_requests"])
            st.caption(f"Complete dataset, plus {plan['warmup_per_model']} warm-ups/model and "
                       f"{plan['repeat_cases']} consistency cases × {max(plan['repetitions']-1,0)} extra rounds.")
    except Exception as exc:
        st.error(str(exc))
    if plan and plan["maximum_requests"]>plan["request_cap"]:
        st.error(f"Full run needs up to {plan['maximum_requests']:,} requests; configured cap is {plan['request_cap']:,}.")
    for model in cfg["models"]:
        if model["name"] in selected:
            issue = bench.model_problem(model)
            if issue:
                st.warning(model.get("display_name",model["name"])+": "+issue+" This model will be skipped.")
    blocked = busy or not selected or not cases or bool(st.session_state["input_errors"]) or plan is None
    full,test = st.columns(2)
    if full.button("Run complete benchmark",type="primary",
                   disabled=blocked or bool(plan and plan["maximum_requests"]>plan["request_cap"])):
        try:
            st.session_state["job"] = RunJob(run_cfg,cases,selected=selected)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if test.button("Test selected models",
                   disabled=blocked or bool(plan and len(selected)>plan["request_cap"])):
        try:
            st.session_state["job"] = RunJob(run_cfg,cases,selected=selected,connection_test=True)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.caption("Short test: one inference request per ticked model, with response status and latency.")
    current_job = st.session_state.get("job")
    if current_job and current_job.finished and current_job.folder and current_job.connection_test:
        try:
            connection_results_view(bench.read_report(current_job.folder))
        except Exception as exc:
            st.error("Cannot load connection test: "+str(exc))

report = None
with results_tab:
    st.subheader("Saved results")
    folders = sorted((BASE/"results").glob("*/manifest.json"),reverse=True)
    paths = [str(p.parent) for p in folders]
    saved = st.session_state.get("report_folder")
    if saved and str(saved) not in paths:
        paths.insert(0,str(saved))
    if paths:
        selected = st.selectbox("Recent run",paths,index=paths.index(str(saved)) if str(saved) in paths else 0,
                                key="recent_run_"+str(saved or ""))
        if st.button("Load selected run",disabled=busy):
            st.session_state["report_folder"] = selected
            st.session_state.pop("report_cache",None)
            st.session_state.pop("job",None)
            st.rerun()
    custom_folder = st.text_input("Or path to an earlier results folder",key="old_folder")
    if st.button("Load this folder",disabled=busy or not custom_folder.strip()):
        st.session_state["report_folder"] = custom_folder
        st.session_state.pop("report_cache",None)
        st.session_state.pop("job",None)
        st.rerun()
    folder = st.session_state.get("report_folder")
    if folder:
        try:
            cache = st.session_state.get("report_cache")
            if not cache or cache[0]!=str(folder):
                cache = (str(folder),bench.read_report(folder))
                st.session_state["report_cache"] = cache
            report = cache[1]
            result_view(report,cfg)
        except Exception as exc:
            st.error("Cannot load report: "+str(exc))
    else:
        st.info("Load a previous run or start a benchmark. Pricing and business scenarios can be changed without making model calls.")
with cases_tab:
    if report:
        case_view(report)
    else:
        st.info("Load a run on Results to inspect original inputs, predictions and full API errors.")

with guide_tab:
    st.subheader("Metrics and how to use them")
    guide = METRIC_GUIDE
    st.dataframe(pd.DataFrame(guide,columns=["Metric","Why it is useful / limitation"]),width="stretch",hide_index=True)
    st.caption("Domain data is essential: taxonomy, policy, conversation history and failure costs change model rankings. The fixture suite checks plumbing; use a reviewed, untouched holdout for a deployment recommendation.")

st.session_state["cfg"] = cfg
