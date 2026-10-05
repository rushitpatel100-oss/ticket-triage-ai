"""Gradio demo: paste a support ticket, get the team, priority and ITIL ticket type.

Run locally:  python app.py   then open http://127.0.0.1:7860
"""

import gradio as gr

from src import config
from src.predict import TicketClassifier

classifier = TicketClassifier()

ITIL_NEXT_STEP = {
    "incident": "Unplanned interruption: restore service as fast as possible, then check whether it is part of a wider problem.",
    "request": "Service request: fulfil it through the standard request process or service catalogue.",
    "problem": "Underlying cause of one or more incidents: open a root-cause investigation.",
    "change": "Change to a service or system: assess risk and approve through change management before work starts.",
}

EXAMPLES = [
    ["Cannot log in after password reset",
     "Hi, I reset my password this morning but the portal still says my credentials are invalid. "
     "I have a client presentation in an hour and need access urgently."],
    ["Invoice charged twice",
     "Our company card was charged twice for the March subscription invoice. Please refund the duplicate payment."],
    ["Request: new laptop for starter",
     "We have a new analyst joining on Monday. Could you set up a laptop with the standard software and email access?"],
    ["Data platform down for all users",
     "Since 09:10 nobody in the team can load dashboards. The page times out with a 503 error. This is blocking all reporting."],
]


def triage(subject: str, body: str):
    if not (subject or body).strip():
        empty = {}
        return empty, empty, empty, "Paste a ticket subject or description first."

    scores = classifier.predict(subject, body)
    top = {task: next(iter(s.items())) for task, s in scores.items()}

    lines = []
    if "queue" in top:
        lines.append(f"**Route to:** {top['queue'][0]}")
    if "priority" in top:
        lines.append(f"**Priority:** {top['priority'][0]}")
    if "type" in top:
        ticket_type = top["type"][0]
        lines.append(f"**ITIL type:** {ticket_type}. {ITIL_NEXT_STEP.get(ticket_type.lower(), '')}")

    unsure = [task for task, (_, p) in top.items() if p < config.REVIEW_THRESHOLD]
    if unsure:
        lines.append(f"**Needs human review:** the model is less than {config.REVIEW_THRESHOLD:.0%} confident "
                     f"about {', '.join(unsure)}.")
    else:
        lines.append("Confident on all predictions, safe to auto-route.")

    backends = sorted({classifier.backend(t) for t in scores})
    lines.append(f"<sub>Model: {', '.join(backends)}</sub>")
    return scores.get("queue", {}), scores.get("priority", {}), scores.get("type", {}), "\n\n".join(lines)


with gr.Blocks(title="Support Ticket Triage") as demo:
    gr.Markdown(
        "# Support Ticket Triage\n"
        "Paste an IT or customer support ticket. A fine-tuned DistilBERT model predicts which team should "
        "handle it, how urgent it is, and its ITIL ticket type. Low-confidence tickets are flagged for a human."
    )
    if not classifier.tasks:
        gr.Markdown("**No trained models found.** Run `python -m src.baseline` or `python -m src.train` first.")

    with gr.Row():
        with gr.Column(scale=3):
            subject = gr.Textbox(label="Subject", placeholder="e.g. VPN keeps disconnecting")
            body = gr.Textbox(label="Description", lines=7, placeholder="Describe the issue...")
            button = gr.Button("Triage ticket", variant="primary")
            gr.Examples(EXAMPLES, inputs=[subject, body])
        with gr.Column(scale=2):
            decision = gr.Markdown()
            queue_out = gr.Label(label="Team (queue)", num_top_classes=3)
            priority_out = gr.Label(label="Priority", num_top_classes=3)
            type_out = gr.Label(label="ITIL ticket type", num_top_classes=4)

    outputs = [queue_out, priority_out, type_out, decision]
    button.click(triage, inputs=[subject, body], outputs=outputs)
    subject.submit(triage, inputs=[subject, body], outputs=outputs)


if __name__ == "__main__":
    demo.launch()
