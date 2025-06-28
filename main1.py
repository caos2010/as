import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ✅ Dashboard chuyên nghiệp, trình bày đẹp
st.set_page_config(page_title="❤️ Phân Tích Sức Khỏe Tim Mạch", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.big-title {font-size:48px; color:#C1121F; text-align:center; margin-bottom:0;}
.sub-title {text-align:center; font-size:20px; color:#555; margin-top:0;}
</style>
<h1 class="big-title">❤️ Ứng Dụng Đánh Giá Nguy Cơ Tim Mạch</h1>
<p class="sub-title">Phân tích dữ liệu sức khỏe và đưa ra đánh giá nguy cơ bệnh tim một cách trực quan</p>
""", unsafe_allow_html=True)

st.write("""---""")

st.sidebar.header("📂 Tải Lên Dữ Liệu Sức Khỏe")
uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])

st.sidebar.info("File CSV cần có cột 'cardio' (0 = bình thường, 1 = có nguy cơ). Bao gồm chỉ số như huyết áp, cân nặng, cholesterol...")

if uploaded_file:
    st.success("✅ Dữ liệu đã tải lên thành công!")
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dữ Liệu Mẫu")
    st.dataframe(df.head(10))

    if 'cardio' not in df.columns:
        st.error("🚫 File không có cột 'cardio'. Hãy kiểm tra dữ liệu!")
    else:
        X = df.drop(columns=[col for col in ['id','cardio'] if col in df.columns])
        y = df['cardio']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner("⏳ Đang huấn luyện mô hình..."):
            grid = GridSearchCV(RandomForestClassifier(), param_grid={"n_estimators": [100]}, cv=5, scoring="f1")
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

        st.markdown("## 📈 Kết Quả Đánh Giá")

        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Độ Chính Xác", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("✅ F1 Score", f"{f1_score(y_test, y_pred):.2f}")
        auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]) if hasattr(grid, 'predict_proba') else None
        if auc is not None:
            col3.metric("📌 AUC", f"{auc:.2f}")
        else:
            col3.warning("🔍 AUC không tính được")

        with st.expander("📑 Tham Số Mô Hình"):
            st.json(grid.best_params_)

        st.success("🎉 Mô hình đã huấn luyện xong! Kết quả đã sẵn sàng.")

else:
    st.info("📌 Vui lòng tải file CSV để bắt đầu phân tích.")
