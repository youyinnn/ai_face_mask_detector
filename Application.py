import torch
from torchvision.io import read_image
import Models
import torchvision.transforms as T
import Training
import matplotlib.pyplot as plt
label_map = {
    0: 'cloth_mask',
    1: 'no_face_mask',
    2: 'surgical_mask',
    3: 'n95_mask',
    4: 'mask_worn_incorrectly',
}


def load_and_preprocess_img(img_path, resize=128):
    transform = T.Compose([T.Resize((resize, resize))])
    image = read_image(img_path)

    return transform(image)


def application_mode(model_type, model_path, img_path):
    model = Training.load_model(model_type, model_path)
    img = load_and_preprocess_img(img_path)
    plt.imshow(img.permute(1, 2, 0))
    img = img.float().reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    prediction = model(img).argmax()
    plt.title("Prediction: "+label_map[prediction.item()])
    plt.show()

    print("Prediction: ", label_map[prediction.item()])


def application_mode_imageset(model_type, model_path, file_path='./data/test_data/'):
    if torch.cuda.is_available():
        # print("Using GPU!")
        device = 'cuda'
    else:
        # print("Using CPU :(")
        device = 'cpu'
    X, y = Training.get_all_data(128, file_path=file_path)
    #X = X.to(device)
    #y = y.to(device)
    model = Training.load_model(model_type, model_path).to(device)
    results = Training.eval_model(model, X, y)
    print(results)
    return results
