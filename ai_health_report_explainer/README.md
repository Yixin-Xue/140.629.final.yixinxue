# AI Health Report Explainer

AI Health Report Explainer is a Streamlit app that uses a pretrained Hugging Face language model to explain medical and public health text in plain language. Users paste health news, a research abstract, a public health report, or a doctor explanation, and the app returns a structured explanation with a summary, risk factors, suggested next steps, limitations, and a concern level.

## Files

- `README.md`: Project description, links, and run instructions.
- `app.py`: Streamlit application code.
- `requirements.txt`: Python packages needed to run the app.
- `sample_inputs/example_health_text.txt`: Example text for testing the app.

## GitHub Repository

Repository link: https://github.com/Yixin-Xue/140.629.final.yixinxue

## How to Run

Install the requirements, then start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run may take extra time because the pretrained model `google/flan-t5-base` must be downloaded from Hugging Face.

## How to Use

Paste an English medical or public health text into the text box, choose the target reader and detail level, and click **Analyze Text**. The output is for education and health communication only. It is not a diagnosis, treatment plan, or replacement for professional medical care.

## Deployed App

Streamlit app: https://140629finalyixinxue.streamlit.app/

## Presentation Video

Video link: https://drive.google.com/file/d/1ucKf1jsxtye34SXOJNTEnqEHdC0d81Tv/view?usp=drive_link
