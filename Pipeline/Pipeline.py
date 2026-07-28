import torch
from multiprocessing import Process, Queue
from torchvision import transforms, datasets
from PIL import Image
import os
import psutil
from tqdm import tqdm
import time
print("-------------✅ libraries ---------------")
torch.backends.quantized.engine = "qnnpack"

# Hybrid models also return a single prediction,
# therefore they use the same inference path.

INFERENCE_MODE = "BASELINE"  # "DMR", "TMR", "BASELINE" Note: TMR and the hybrid both use the TMR keyword :)

# True: images first pass through the modality classifier.
# False: images are directly routed to their diagnosis unit.

USE_CLASSIFIER = True  


# ------------------ DMR --------------------------
# colon_model = "/home/amy_rpi/Desktop/FT/Colon/Models/DMR_Colon_RN18.pt"
# mri_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/MRI/DMR_MRI_RN18.pt"
# xray_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/Xray/DMR_Xray_RN18.pt"

# ------------------ Hybrid --------------------------
# 
# colon_model = "/home/amy_rpi/Desktop/FT/Colon/Models/hybrid_Colon_RN18.pt"
# mri_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/MRI/hybrid_MRI_RN18.pt"
# xray_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/Xray/hybrid_Xray_RN18.pt"

# ------------------ TMR --------------------------
# colon_model = "/home/amy_rpi/Desktop/FT/Colon/Models/TMR_Colon_RN18.pt"
# mri_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/MRI/TMR_MRI_RN18.pt"
# xray_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/Xray/TMR_Xray_RN18.pt"
# ------------------ Baseline --------------------------
colon_model = "/home/amy_rpi/Desktop/FT/Colon/Models/quantized_jit_Colon_RN18_M1(NP)_int8.pt"
mri_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/MRI/quantized_scripted_MRI_RN18_M1(NP_int8).pt"
xray_model = "/home/amy_rpi/Desktop/FT/MRI&Xray/Models/Xray/quantized_jit_scripted_RN18_Xray_M2_(NP_int8).pt"


colon_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gray_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
print("-------------✅ Transformrs -------------")
def get_class_names(test_dir):
    return datasets.ImageFolder(test_dir).classes

colon_classes = get_class_names("/home/amy_rpi/Desktop/Test datasets/Colon_Test/test")
xray_classes  = get_class_names("/home/amy_rpi/Desktop/Test datasets/Xray_Test/test")
mri_classes   = get_class_names("/home/amy_rpi/Desktop/Test datasets/MRI_Test/test")
print("---------------✅ Classes----------------")
def load_image(path, transform):
    img = Image.open(path)
    img = transform(img)
    return img.unsqueeze(0)

def run_model(name, model_path, transform, class_names, q, result_q, core):

    torch.set_num_threads(1)
    p = psutil.Process()
    p.cpu_affinity([core])

    print(f"[{name}] PID: {os.getpid()} | Core: {p.cpu_num()}")

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    for path in tqdm(iter(q.get, None), desc=f"{name} Processing"):

        img = load_image(path, transform)

        with torch.no_grad():
            output = model(img)


        if INFERENCE_MODE == "DMR":

            y_pred = output
            b1 = y_pred["b1"].view(-1)
            agreement = y_pred["agreement"].view(-1)

            pred = int(b1.item())
            is_agree = int(agreement.item())

            pred_label = class_names[pred]

            result = f"{name} → {pred_label} | Agree: {is_agree}"

        elif INFERENCE_MODE == "TMR":

            pred = int(output.view(-1).item())
            pred_label = class_names[pred]

            result = f"{name} → {pred_label}"

        elif INFERENCE_MODE == "BASELINE":

            prob = torch.sigmoid(output).view(-1).item()
            pred = 1 if prob > 0.5 else 0

            pred_label = class_names[pred]

            result = f"{name} → {pred_label} (p={prob:.3f})"


        result_q.put(result)



def classifier_process(paths, colon_q, mri_q, xray_q):

    p = psutil.Process()
    p.cpu_affinity([0])

    model = torch.jit.load(
        "/home/amy_rpi/Desktop/FT/Simple_CNN_pruned_jit_45%_int8.pt",
        map_location="cpu"
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    mapping = {0: "colon", 1: "mri", 2: "xray"}
    
    for path in tqdm(paths, desc="Classifying"):

        img = load_image(path, transform)

        with torch.no_grad():
            pred = torch.argmax(model(img)).item()

        target = mapping[pred]

        if target == "colon":
            colon_q.put(path)
        elif target == "mri":
            mri_q.put(path)
        else:
            xray_q.put(path)


    colon_q.put(None)
    mri_q.put(None)
    xray_q.put(None)



def main():

    if USE_CLASSIFIER:
        dataset_path = "/home/amy_rpi/Desktop/Test datasets/Mixed_Test_all"

        image_paths = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    else:
        image_paths = []

        for key, path in {
            "colon": "/home/amy_rpi/Desktop/Test datasets/Colon_Test/test",
            "xray":  "/home/amy_rpi/Desktop/Test datasets/Xray_Test/test",
            "mri":   "/home/amy_rpi/Desktop/Test datasets/MRI_Test/test"
        }.items():

            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, f))

    print(f"Total images: {len(image_paths)}")

    colon_q, mri_q, xray_q = Queue(), Queue(), Queue()
    result_q = Queue()

  
    colon_p = Process(target=run_model,
                      args=("Colon", colon_model, colon_transform, colon_classes, colon_q, result_q, 1))

    mri_p = Process(target=run_model,
                    args=("MRI", mri_model, gray_transform, mri_classes, mri_q, result_q, 2))

    xray_p = Process(target=run_model,
                     args=("Xray", xray_model, gray_transform, xray_classes, xray_q, result_q, 3))

    colon_p.start()
    mri_p.start()
    xray_p.start()

    milestones = [1, 10, 100, 500, 1000,1500,2000,2010]
    milestone_times = {}

    start = time.time()

    if USE_CLASSIFIER:
        classifier = Process(target=classifier_process,
                             args=(image_paths, colon_q, mri_q, xray_q))
        classifier.start()
    else:

        for path in image_paths:

            if "Colon" in path:
                colon_q.put(path)
            elif "MRI" in path:
                mri_q.put(path)
            else:
                xray_q.put(path)

        colon_q.put(None)
        mri_q.put(None)
        xray_q.put(None)


    expected = len(image_paths)
    received = 0

    while received < expected:
        result_q.get()
        received += 1
        if received in milestones:
            milestone_times[received] = time.time() - start

    end = time.time()
    
    print("\n--- Milestone Times ---")

    for n in milestones:
        if n in milestone_times:
            print(f"{n} images -> {milestone_times[n]:.4f} sec")

    print("\n✅ DONE")
    print(f"Mode: {'WITH classifier' if USE_CLASSIFIER else 'WITHOUT classifier'}")
    print(f"Total Time: {end - start:.4f} sec")
    print(f"Avg Time/Image: {(end - start)/expected:.6f} sec")

print("----------------------------✅ Start ! ---------------------------------")

if __name__ == "__main__":
    main()

