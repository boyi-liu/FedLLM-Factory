import streamlit as st
import pandas as pd
import time
import os
from utils.options import args_parser
from main import FedSim

# --- 1. Page Configuration ---
st.set_page_config(page_title="FedLLM Factory", layout="wide")
st.title("🚀 FedLLM Factory")

# --- 2. Sidebar Parameter Configuration ---
st.sidebar.header("Args for training")
args = args_parser()

# Use Streamlit parameter inputs to override args
alg_list = [
    'fedit',
    'ffalora',
    'flora',
    'flexlora',
    'feddpa',
    'fedsalora',
    'fedexlora',
    'fedsvd',
    'ravan',
    'rolora',
    'fedlease',
    'slora',
    'ilora',
    'fedrotlora',
    'hilora',
    'fedmomentum'
]
dataset_list = [
    'sst2',
    'imdb',
    'dolly',
    'gsm8k'
]

args.alg = st.sidebar.selectbox("Algorithm", alg_list, index=0)
args.dataset = st.sidebar.selectbox("Dataset", dataset_list, index=0)
args.rnd = st.sidebar.number_input("Communication Round", value=args.rnd, min_value=1)
args.cn = st.sidebar.slider("Client Scale", 1, 100, args.cn)
args.lr = st.sidebar.number_input("Learning Rate", value=args.lr, format="%.4f")
args.test_gap = st.sidebar.number_input("Test Gap", value=args.test_gap, min_value=1)

# --- 3. Core Function ---
class StreamlitFedSim(FedSim):
    def __init__(self, args, progress_bar, status_text, chart_placeholder):
        super().__init__(args)
        self.pb = progress_bar
        self.st_text = status_text
        self.chart_placeholder = chart_placeholder
        self.history = []

    def simulate(self):
        TEST_GAP = self.args.test_gap
        try:
            for rnd in range(self.args.rnd):
                progress = (rnd + 1) / self.args.rnd
                self.pb.progress(progress)
                self.st_text.text(f"Round {rnd}...")

                self.server.round = rnd
                self.server.run()

                if (self.args.rnd - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    
                    ret_dict['round'] = rnd
                    self.history.append(ret_dict)
                    df_history = pd.DataFrame(self.history).set_index('round')
                    self.chart_placeholder.line_chart(df_history) 
                    
                    st.toast(f"Round {rnd} Test Passed!")
        except Exception as e:
            st.error(f"Error: {e}")

# --- 4. Running Button ---
if st.sidebar.button("Start Running", type="primary"):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Progress")
        p_bar = st.progress(0)
        status_msg = st.empty()
    
    with col2:
        st.subheader("Metrics")
        chart_space = st.empty()

    sim = StreamlitFedSim(args, p_bar, status_msg, chart_space)
    sim.simulate()
    st.success("Finished all rounds!")