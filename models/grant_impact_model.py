import streamlit as st
import pandas as pd

def render_grant_impact_model(df):
    """Render the grant impact classification model"""
    
    st.markdown("### 🤖 Low-Impact Grant Classification Model")
    model_cols = st.columns([3, 1])

    with model_cols[0]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import classification_report

            # Prepare data for modeling
            model_df = df.dropna(subset=["value", "savings", "agency", "recipient"])
            
            if model_df.empty:
                st.warning("⚠️ Not enough data to train the model.")
                return
            
            # Create target variable (low impact grants)
            model_df = model_df.copy()
            model_df["is_low_impact"] = ((model_df["savings"] == 0) | (model_df["value"] < 10000)).astype(int)

            # Encode categorical variables
            le_agency = LabelEncoder()
            le_recipient = LabelEncoder()
            model_df["agency_enc"] = le_agency.fit_transform(model_df["agency"].astype(str))
            model_df["recipient_enc"] = le_recipient.fit_transform(model_df["recipient"].astype(str))

            # Prepare features and target
            X = model_df[["value", "agency_enc", "recipient_enc"]]
            y = model_df["is_low_impact"]

            if len(model_df) < 2:
                st.warning("⚠️ Not enough data to train the model.")
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            if X_train.empty or y_train.empty:
                st.warning("⚠️ Not enough data after train/test split.")
                return

            # Train model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display results
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("Classification Report:")
            st.dataframe(report_df)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': ['Grant Value', 'Agency', 'Recipient'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.write("Feature Importance:")
            st.dataframe(feature_importance)

        except ImportError:
            st.error("⚠️ scikit-learn is required for the model. Please install it with: pip install scikit-learn")
        except Exception as e:
            st.error(f"⚠️ Model training failed: {e}")

    with model_cols[1]:
        with st.container(border=True):
            st.markdown("#### 🧠 Model Explanation")
            st.markdown(
                '''
                This Random Forest model identifies potentially low-impact grants,
                defined as those with zero reported savings or a grant value under $10,000.
                It uses agency and recipient identifiers along with funding value.
                Use this model to investigate patterns and refine grant effectiveness.
                '''
            )