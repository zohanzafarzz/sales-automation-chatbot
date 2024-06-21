from rest_framework import serializers
from django.core.validators import RegexValidator

class MessageSerializer(serializers.Serializer):
    message = serializers.CharField(
        max_length=500,
        validators=[
            RegexValidator(
                regex='^[a-zA-Z0-9\s,.!?-]*$',
                message='Message contains invalid characters.',
                code='invalid_message'
            )
        ]
    )
