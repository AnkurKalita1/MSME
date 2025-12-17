# DynamoDB Local Setup

The WBS API requires DynamoDB Local to be running to load training data.

## Installation

### macOS (using Homebrew)
```bash
brew install dynamodb-local
```

### Manual Installation
1. Download DynamoDB Local from: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.DownloadingAndRunning.html
2. Extract the archive
3. Run: `java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -port 8000`

## Starting DynamoDB Local

### Using Homebrew
```bash
dynamodb-local -port 8000
```

### Manual
```bash
java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -port 8000
```

## Verify It's Running

```bash
curl http://localhost:8000/
```

You should see a response (even if it's an error, it means DynamoDB is running).

## Loading Data

Once DynamoDB Local is running, you need to load the BOQ/WBS data. Run this command from the `backend/python` directory:

```bash
cd backend/python
python3 wbs_api.py upload ../Datas
```

This will:
- Create the `MSME_BOQ_WBS` table automatically
- Load 500 records from `synthetic_boq_wbs_unique.csv`
- Make the data available for WBS generation

**Note:** You only need to run this once. The data persists in DynamoDB Local until you stop it.

## Note

- DynamoDB Local must be running before starting the backend server
- Keep the DynamoDB Local terminal window open while using the application
- The default port is 8000 (as configured in the code)

