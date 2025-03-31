import boto3

dynamo_client = boto3.client("dynamodb")


if __name__ == "__main__":
    dynamo_client.create_table(
        AttributeDefinitions=[
            {"AttributeName": "id", "AttributeType": "S"},
            {"AttributeName": "email", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
        ],
        KeySchema=[
            {
                "AttributeName": "id",
                "KeyType": "HASH",
            },
            {
                "AttributeName": "timestamp",
                "KeyType": "RANGE",
            },
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "EmailIndex",
                "KeySchema": [
                    {"AttributeName": "email", "KeyType": "HASH"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        TableName="contact",
        BillingMode="PAY_PER_REQUEST",
    )
