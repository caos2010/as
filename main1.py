import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="❤️ Dashboard Đánh Giá Tim Mạch", page_icon="✅", layout="wide")

# =====================
# Tiêu đề + phong cách
# =====================
st.markdown("""
<style>
.big-title {font-size:50px; color:#C1121F; text-align:center; margin-bottom:0;}
.sub-title {text-align:center; font-size:22px; color:#333; margin-top:0;}
.metric-title {font-weight:bold; font-size:18px;}
</style>
<h1 class="big-title">❤️ Hệ Thống Đánh Giá Nguy Cơ Tim Mạch</h1>
<p class="sub-title">Phân tích sức khỏe và đưa ra đánh giá nguy cơ bệnh tim một cách trực quan, chi tiết</p>
""", unsafe_allow_html=True)

st.write("""---""")

# =====================
# Sidebar rõ ràng
# =====================
st.sidebar.header("📂 Tải Lên Dữ Liệu Sức Khỏe")
uploaded_file = st.sidebar.file_uploader("Chọn tệp CSV", type=["csv"])

st.sidebar.info("⚙️ File cần có cột `cardio` (0 = bình thường, 1 = có nguy cơ).\nBao gồm các chỉ số: huyết áp, cân nặng, cholesterol,... để kết quả chính xác.")

# =====================
# Phân tích dữ liệu
# =====================
if uploaded_file:
    st.success("✅ Dữ liệu đã được tải lên thành công!")
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dữ Liệu Nhập Vào")
    st.dataframe(df.head(10))

    if 'cardio' not in df.columns:
        st.error("🚫 File không có cột `cardio`. Hãy kiểm tra lại.")
    else:
        features = df.drop(columns=[col for col in ['id', 'cardio'] if col in df.columns])
        target = df['cardio']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        with st.spinner("⏳ Đang huấn luyện mô hình Random Forest..."):
            param_grid = {
                "n_estimators": [100],
                "max_depth": [None],
                "min_samples_split": [2],
                "min_samples_leaf": [1]
            }
            model = RandomForestClassifier(random_state=42)
            grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)

        st.markdown("<h3 style='text-align:center;'>📈 Kết Quả Đánh Giá</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="🎯 Độ Chính Xác", value=f"{accuracy_score(y_test, y_pred):.2f}")
            st.metric(label="✅ Precision", value=f"{precision_score(y_test, y_pred):.2f}")
        with col2:
            st.metric(label="⚖️ F1 Score", value=f"{f1_score(y_test, y_pred):.2f}")
            st.metric(label="📌 Recall", value=f"{recall_score(y_test, y_pred):.2f}")
        with col3:
            st.metric(label="🔍 AUC", value=f"{roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.2f}")

        with st.expander("📑 Tham Số Mô Hình Chi Tiết"):
            st.json(grid.best_params_)

        st.success("🎉 Hoàn thành huấn luyện! Kết quả phân tích đã sẵn sàng.")

else:
    st.info("📥 Vui lòng tải file CSV để hệ thống bắt đầu đánh giá.")

st.write("---")

st.markdown("""
💡 **Lưu ý:** Dữ liệu càng đầy đủ và chính xác thì kết quả đánh giá càng đáng tin cậy.
📌 **Hãy chuẩn bị dữ liệu sức khỏe của bạn một cách cẩn thận!**
""")
