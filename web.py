import gradio as gr
from service import Service


def doctor_bot(message, history):
    service = Service()
    return service.answer(message, history)


css = '''
.gradio-container { max-width:850px !important; margin:20px auto !important;}
.message { padding: 10px !important; font-size: 14px !important;}
'''

demo = gr.ChatInterface(
    css=css,
    fn=doctor_bot,
    title='医疗问诊机器人',
    chatbot=gr.Chatbot(height=400, bubble_full_width=False),
    theme=gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="请输入您的问题", container=False, scale=7),
    examples=['你好。', '寻医问药网的客服电话是多少？', '鼻炎是一种什么病？',
              '一般会有哪些症状？', '黄金树之影什么时候上线？'],
    submit_btn=gr.Button('提交', variant='primary'),
    clear_btn=gr.Button('清空记录'),
    retry_btn=None,
    undo_btn=None,
)

if __name__ == '__main__':
    demo.launch(share=False)
