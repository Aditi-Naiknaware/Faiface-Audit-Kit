import gradio as gr
import torch, json
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

from src.config import CFG
from src.model import build_model
from src.utils import AGE_IDX2STR, RACE_IDX2STR, GENDER_IDX2STR

cfg = CFG()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
model_base_path = Path("models/model_reweighted.pt") if Path("models/model_reweighted.pt").exists() else Path("models/model_best.pt")
model_mitig_path = Path("models/model_groupdro.pt")

model_base = build_model(cfg.model_name, num_classes=9).to(device)
model_base.load_state_dict(torch.load(model_base_path, map_location=device))
model_base.eval()

model_mitig = build_model(cfg.model_name, num_classes=9).to(device)
model_mitig.load_state_dict(torch.load(model_mitig_path, map_location=device))
model_mitig.eval()

tfm = T.Compose([
    T.Resize((cfg.img_size, cfg.img_size)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict(pil_img: Image.Image, which:str):
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        if which=="Baseline":
            logits = model_base(x)
        else:
            logits = model_mitig(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    # return top-3
    top3 = sorted(list(enumerate(probs)), key=lambda z: z[1], reverse=True)[:3]
    return {AGE_IDX2STR[i]: float(p) for i,p in top3}

# Load saved reports (show side-by-side)
def load_report(path):
    if not Path(path).exists(): return None
    with open(path) as f: return json.load(f)

rep_base = load_report("reports/reweighted_report.json") or load_report("reports/baseline_report.json")
rep_mitig = load_report("reports/groupdro_report.json")

def report_table():
    rows = []
    def tidy(rep, name):
        if not rep: return
        rows.append([name, "overall_acc", rep["overall_acc"]])
        rows.append([name, "worst_group", rep["worst_group"]])
        rows.append([name, "worst_group_acc", rep["per_group_acc"][rep["worst_group"]]])
        rows.append([name, "acc_gap", rep["acc_gap"]])
        if "ece" in rep:
            rows.append([name, "ece", rep["ece"]])
    tidy(rep_base, "Baseline/Reweighted")
    tidy(rep_mitig, "GroupDRO")
    return rows

with gr.Blocks() as demo:
    gr.Markdown("# FairFace Audit Kit — Demo")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload a face image")
            which = gr.Radio(choices=["Baseline","GroupDRO"], value="Baseline", label="Model")
            out = gr.Label(num_top_classes=3, label="Top-3 age predictions")
            btn = gr.Button("Predict")
        with gr.Column():
            gr.Markdown("### Saved Metrics")
            tbl = gr.Dataframe(headers=["run","metric","value"], value=report_table(), wrap=True)
            gr.Markdown("See `reports/` for full results & plots.")

    btn.click(fn=predict, inputs=[inp, which], outputs=out)

# In JupyterHub, inline=True gives an iframe preview; fallback to share=True if proxy blocks it.
if __name__ == "__main__":
    import gradio as gr
    gr.close_all()  # free any old servers/ports
    demo.launch(
        share=True,            # <-- prints https://xxxx.gradio.live
        server_name="0.0.0.0",
        server_port=None,      # auto-pick a free port
        inline=False,          # don't try to iframe inside Jupyter
        show_error=True,
    )