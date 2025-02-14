from fastapi import FastAPI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import initialize_agent,AgentType
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from pathlib import Path
from fastapi.responses import Response
import os
import uvicorn
import matplotlib.pyplot as plt
import pandas as pd
import re
import io
import matplotlib

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = FastAPI()
llm = ChatGroq(model="gemma2-9b-it")

matplotlib.use('Agg')

parent_path = Path(__file__).parent
data_file_path = parent_path / "MSA_SALES_DETAILS_STORE1AND2.xlsx"

# Load Data
sales_data = pd.read_excel(data_file_path)

def generate_plot_code(query):
    prompt = f"""
    Generate **ONLY** valid Python code for Matplotlib.  
    **Do not include any explanations or markdown.**  
    **Dataset:** `sales_data`  
    Columns: ['Amount', 'Day', 'Month', 'Store_num', 'nacs_cat_description', 'size', 'sku_description']  
    **Rules:**  
    - Use `sales_data` only.  
    - If "store" is in the query, use `Store_num`. 
    - The column 'Month' contains numeric data such as 1 for January, 2 for February , 3 for March as so on.. use **ONLY** dataset 'sales_data' as your reference to generate code
    - No dummy data, no fake columns.  
    - Use `plt.show()` for display.  
    - **Return only Python code**.  
    **Example Query:** "Show total sales per store."
    ```python
    import matplotlib.pyplot as plt
    store_sales = sales_data.groupby('Store_num')['Amount'].sum()
    store_sales.plot(kind='bar')
    plt.xlabel('Store Number')
    plt.ylabel('Total Sales Amount')
    plt.title('Total Sales by Store')
    plt.show()
    ```
    **Now generate the code for:** "{query}"
    """
    return llm.predict(prompt)

generate_code_tool = Tool(
    name="GeneratePlotCode",
    func=generate_plot_code,
    description="Generates Python code for plotting graphs from user queries."
)

agent = initialize_agent(
    tools=[generate_code_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@app.post("/plotGraphByUserPrompt")
def plotGraphByUserPrompt(prompt:str):
    if prompt:
        try:            
            response = agent.run(prompt)

            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            cleaned_code = code_match.group(1).strip() if code_match else response.strip()
            fig, ax = plt.subplots()
            exec(cleaned_code, globals(), {"sales_data": sales_data, "plt": plt, "ax": ax}) 
            
            img_io = io.BytesIO()
            plt.savefig(img_io,format="png")
            plt.close(fig)
            img_io.seek(0)
            #print(len(img_io.getvalue()))
            
            return Response(content=img_io.getvalue(), media_type="image/png")

        except Exception as e:
           return f"Error in processing : {str(e)}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #plotGraphByUserPrompt("plot graph for top 3 selling products in store 1 in january")