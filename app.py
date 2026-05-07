import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="Gold Predictor AI", page_icon="⛏️", layout="wide")

st.title("🗺️ Интеллектуальная система прогноза золотого оруденения")
st.markdown("Интерактивный инструмент пространственного выделения рудных узлов на базе ансамблевой модели CatBoost.")

# 1. Загрузка модели в кэш
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('gold_predictor_prod.cbm')
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Модель CatBoost загружена")
except Exception as e:
    st.sidebar.error("❌ Ошибка загрузки: убедитесь, что 'gold_predictor_prod.cbm' находится в той же папке.")

# 2. Боковая панель: Загрузка файла и настройки
uploaded_file = st.sidebar.file_uploader("📂 Загрузите данные (CSV или Excel)", type=['csv', 'xlsx'])
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Порог уверенности алгоритма", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

if uploaded_file:
    try:
        # Чтение файла
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Кнопка запуска
        if st.button("🚀 Рассчитать прогноз перспективности", type="primary"):
            with st.spinner('Анализ комплекса геохимических и спутниковых данных...'):
                
                # Поиск географических координат
                lon_col, lat_col = None, None
                for col in df.columns:
                    if col.upper() in ['LON', 'LONGITUDE', 'X_GRAD']: lon_col = col
                    if col.upper() in ['LAT', 'LATITUDE', 'Y_GRAD']: lat_col = col
                
                if not lon_col or not lat_col:
                    st.warning("⚠️ Координаты (Lon/Lat) не найдены. Карта будет построена без привязки к рельефу.")
                    lon_col, lat_col = 'X', 'Y'
                    use_mapbox = False
                else:
                    use_mapbox = True

                # Очистка от служебных колонок и прогнозирование
                drop_cols = ['AU', 'Target', 'spatial_cluster', 'system:index', 'X', 'Y', 
                             'POINT_X', 'POINT_Y', 'Lon', 'Lat', 'X_GRAD', 'Y_GRAD', 'x_grad', 'y_grad', 'point_id', 'POINT_ID']
                features = [col for col in df.columns if col not in drop_cols]
                
                X_predict = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
                probs = model.predict_proba(X_predict)[:, 1]
                df['Probability'] = np.round(probs, 4)
                
                # Фильтрация прогноза по ползунку
                df_filtered = df[df['Probability'] >= threshold]

                st.markdown(f"### Выделенные аномалии (Порог вероятности ≥ {threshold})")
                st.write(f"Локализовано высокоперспективных точек: **{len(df_filtered)}** из {len(df)}")
                st.markdown("---")
                
                # ==========================================
                # ОТРИСОВКА КАРТЫ НА ВЕСЬ ЭКРАН
                # ==========================================
                map_style = "white-bg" # Отключаем базовую векторную карту
                
                if not df_filtered.empty:
                    if use_mapbox:
                        fig_pred = px.scatter_mapbox(
                            df_filtered, lat=lat_col, lon=lon_col, color='Probability',
                            color_continuous_scale='Turbo', size_max=12, zoom=8, opacity=0.85,
                            mapbox_style=map_style, range_color=[0, 1], 
                            hover_data=['Probability'] + features[:3]
                        )
                        
                        # Добавляем бесплатный спутниковый слой высокого разрешения от Esri
                        fig_pred.update_layout(
                            height=450, 
                            margin={"r":0,"t":0,"l":0,"b":0},
                            mapbox_layers=[
                                {
                                    "below": 'traces',
                                    "sourcetype": "raster",
                                    "sourceattribution": "Esri",
                                    "source": [
                                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                                    ]
                                }
                            ]
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        fig_pred = px.scatter(df_filtered, x=lon_col, y=lat_col, color='Probability', color_continuous_scale='Turbo', range_color=[0, 1])
                        fig_pred.update_layout(height=500)
                        st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    st.warning("При таком высоком пороге рудных узлов не найдено. Снизьте порог в левом меню.")
                
                # Скачивание результата
                st.markdown("---")
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать координаты найденных аномалий (CSV)", 
                    data=csv, file_name="gold_predictive_anomalies.csv", mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Произошла ошибка при обработке: {e}")
