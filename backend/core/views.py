import numpy as np
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .utils import predict_external_image
import tempfile
import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.decorators import api_view
@method_decorator(csrf_exempt, name='dispatch')
class FoodRecogniseView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        img_file = request.FILES['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            for chunk in img_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        try:
            class_name, confidence = predict_external_image(tmp_path)
            # DEBUG: in ra giá trị nhận được
            print("Prediction:", class_name)
            print("Confidence:", confidence)
        except Exception as e:
            os.remove(tmp_path)
            print("Prediction error:", str(e))
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Kiểm tra confidence có phải số hợp lệ không
        if confidence is None or np.isnan(confidence):
            confidence = 0.0
            class_name = "Unknown"

  

        return Response({
            "class_name": class_name,
            "confidence": float(confidence),
        })


@api_view(['GET'])
def draw_view(request):
    return render(request, 'draw.html')