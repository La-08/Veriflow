import py_compile
files = ['VERIFLOW/main.py','VERIFLOW/llm/deepseek_client.py']
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('ok', f)
    except Exception as e:
        print('err', f, e)
