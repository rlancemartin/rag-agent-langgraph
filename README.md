# rag-agent-langgraph

Follow steps [here](https://langchain-ai.github.io/langgraph/cloud/quick_start/#deploy-from-github-with-langgraph-cloud):

Test basic functionality locally -
```
pip install -U langgraph-cli
langgraph test
```

Run -
```
curl --request POST \
    --url http://localhost:8123/runs/stream \
    --header 'Content-Type: application/json' \
    --data '{
    "assistant_id": "rag_agent",
    "input": {"question": "how does agent memory work?","steps": []},
    "metadata": {},
    "config": {
        "configurable": {}
    },
    "multitask_strategy": "reject",
    "stream_mode": [
        "values"
    ]
}'
```





