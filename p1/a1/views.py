from django.http import JsonResponse
import joblib
import json
import os
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'test_ml_api.pkl'))
@csrf_exempt
def predict(req):
    if req.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'},status=405)
    try:
        data = json.loads(req.body)
        hours = float(data['values'])
        result = model.predict([[hours]])
        return JsonResponse({"input": hours,"Predicted Score": round(float(result[0]), 2)})

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return JsonResponse({'error': 'Send valid JSON with a numeric "values" field'},status=400)
