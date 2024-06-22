from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import MessageSerializer
from .models import generate_response

class ChatbotView(APIView):

    def post(self, request):
        serializer = MessageSerializer(data=request.data)
        if serializer.is_valid():
            sanitized_message = serializer.validated_data['message']
            chatbot_response = generate_response(sanitized_message)
            clean_chatbot_response = chatbot_response.replace("**", "")
            # print(clean_chatbot_response)
            return Response({'response': clean_chatbot_response}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)