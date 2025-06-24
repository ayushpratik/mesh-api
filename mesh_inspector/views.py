from django.http import JsonResponse
from rest_framework.decorators import api_view

from mesh_django import settings
from .utils import process_image
import os
import cv2

# @api_view(['POST'])
# def inspect_mesh(request):
#     image = request.FILES.get('image')
#     weight_class = request.POST.get('weight_class')
    

#     image_path = os.path.join("images", image.name)

#     img = cv2.imread(image_path)
#     ruler_pixels = img[500:510, 100:360]
#     print("Width in pixels:", ruler_pixels.shape[1])
#     scale=float(ruler_pixels.shape[1])
#     os.makedirs("images", exist_ok=True)
#     with open(image_path, 'wb+') as f:
#         for chunk in image.chunks():
#             f.write(chunk)

#     results, output_path = process_image(image_path, weight_class, scale)
#     return JsonResponse({
#         "measurements": results,
#         "annotated_image": output_path
#     })

@api_view(['POST'])
def inspect_mesh(request):
    image = request.FILES.get('image')
    weight_class = request.POST.get('weight_class')

    # Save the uploaded image first
    print("Path = ",settings.BASE_DIR)
    images_dir = os.path.join(settings.BASE_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)

    image_path = os.path.join(images_dir, image.name)
    with open(image_path, 'wb+') as f:
        for chunk in image.chunks():
            f.write(chunk)

    # Now read the saved image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return JsonResponse({'error': 'Image failed to load'}, status=400)

    # Safely extract a known 1-inch segment
    try:
        ruler_pixels = img[500:510, 100:360]
        scale = float(ruler_pixels.shape[1])  # This means 260px = 1 inch
    except Exception as e:
        return JsonResponse({'error': f'Failed to extract ruler: {str(e)}'}, status=400)

    # Process mesh detection
    results, output_path = process_image(image_path, weight_class, scale)

    return JsonResponse({
        "measurements": results,
        "annotated_image": output_path
    })