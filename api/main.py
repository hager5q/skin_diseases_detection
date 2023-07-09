import torch
import torchvision
from PIL import Image
from fastapi import FastAPI, File
from fastapi import applications
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import io



pre_process = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization layer
])

model = torch.jit.load("convnext.pt")
softmax_layer = torch.nn.Softmax(dim=1) # define softmax


app = FastAPI(
    title="Skin Classifier",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory="./static"), name="static")
def swagger_monkey_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
    )
applications.get_swagger_ui_html = swagger_monkey_patch
    
@app.post("/classify")
def get_classification_result(file: bytes = File(...)):
    """Get predicted class of skin lesion"""
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    tensor = pre_process(input_image)
    tensor = torch.unsqueeze(tensor,0)
    out = model(tensor)
    out = softmax_layer(out)
    class_id = out.argmax(-1).item()
    score = out[0][class_id].item()
    print(out)
    print(score)
    print(class_id)
    
    return {"class_id":class_id,"score":score}
