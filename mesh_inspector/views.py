from django.http import JsonResponse
from rest_framework.decorators import api_view
from .utils import process_image
import os
import cv2

@api_view(['POST'])
def inspect_mesh(request):
    image = request.FILES.get('image')
    weight_class = request.POST.get('weight_class')
    

    image_path = os.path.join("images", image.name)

    img = cv2.imread(image_path)
    ruler_pixels = img[500:510, 100:360]
    print("Width in pixels:", ruler_pixels.shape[1])
    scale=float(ruler_pixels.shape[1])
    os.makedirs("images", exist_ok=True)
    with open(image_path, 'wb+') as f:
        for chunk in image.chunks():
            f.write(chunk)

    results, output_path = process_image(image_path, weight_class, scale)
    return JsonResponse({
        "measurements": results,
        "annotated_image": output_path
    })
