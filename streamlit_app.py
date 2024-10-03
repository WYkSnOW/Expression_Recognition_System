import streamlit as st
import pandas as pd

# ------ PART 1 ------

df = pd.DataFrame(
    [
       {"command": "st.selectbox", "rating": 4, "is_widget": True},
       {"command": "st.balloons", "rating": 5, "is_widget": False},
       {"command": "st.time_input", "rating": 3, "is_widget": True},
   ]
)

st.title('Facial Expression Recognition System Using Machine Learning')

# Display text

st.text('Team Members: Waiyuk Kwong, Zhihui Chen, Tyler Lin, Blane Richard York, Carter Daniel Robinson')

st.header('Introduction/Background')
st.write('Facial expressions play a crucial role in non-verbal communication, conveying emotions and intentions.')

st.header('Problem Definition')

st.header('Methods')

st.header('Results and Discussion')

st.header('References')
st.markdown("""
1. A. Huang and M. J. C. Samonte, "Using Emotion Analysis, Eye tracking, and Head Movement to Monitor Student Engagement among ESL Students with Facial Recognition Algorithm (Mediapipe)," 2024 7th International Conference on Advanced Algorithms and Control Engineering (ICAACE), Shanghai, China, 2024, pp. 509-513, doi: [10.1109/ICAACE61206.2024.10548871](https://doi.org/10.1109/ICAACE61206.2024.10548871).
2. F. M. Sadikoğlu and M. Idle Mohamed, "Facial Expression Recognition Using CNN," 2022 International Conference on Artificial Intelligence in Everything (AIE), Lefkosa, Cyprus, 2022, pp. 95-99, doi: [10.1109/AIE57029.2022.00025](https://doi.org/10.1109/AIE57029.2022.00025).
3. K. Wang, R. Li, and L. Zhao, "Real-time facial expressions recognition system for service robot based-on ASM and SVMs," 2010 8th World Congress on Intelligent Control and Automation, Jinan, 2010, pp. 6637-6641, doi: [10.1109/WCICA.2010.5554164](https://doi.org/10.1109/WCICA.2010.5554164).
""")


st.markdown('_**Markdown**_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')

# * optional kwarg unsafe_allow_html = True

# Interactive widgets
st.button('Hit me')
st.data_editor(df)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')

# -- add download button (start) --
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)
# -- add download button (end) --

st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')

# ------ PART 2 ------

data = pd.read_csv("employees.csv")

# Display Data
st.dataframe(data)
st.table(data.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
st.metric('My metric', 42, 2)

# Media
st.image('./smile.png')

# Display Charts
st.area_chart(data[:10])
st.bar_chart(data[:10])
st.line_chart(data[:10])
# st.map(data[:10])
st.scatter_chart(data[:10])

# Add sidebar
a = st.sidebar.radio('Select one:', [1, 2])
st.sidebar.caption("This is a cool caption")
st.sidebar.image('./smile.png')

# Add columns
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")
