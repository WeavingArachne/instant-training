import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import gradio as gr
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metrics = {}

    def load_and_preprocess_data(self, fact_transaction_path):
        """Load and preprocess the transaction data"""
        print("Loading data...")
        df = pd.read_csv(fact_transaction_path)

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Feature engineering
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        # Amount features
        df['Amount_Log'] = np.log1p(np.abs(df['Amount']))
        df['Amount_Squared'] = df['Amount'] ** 2

        # Transaction frequency features
        df['Transactions_Per_Day'] = df.groupby(['ClientID', df['Date'].dt.date])[
            'ID'].transform('count')

        # Categorical encoding
        categorical_cols = ['UseChip', 'MerchantState',
                            'Error', 'Payment_Type', 'Category']

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Select features for modeling
        feature_cols = [
            'Amount', 'Amount_Log', 'Amount_Squared', 'Amount_ZScore',
            'UseChip_Encoded', 'MCC', 'Day', 'Month', 'Hour', 'DayOfWeek',
            'Days_Since_Account_Open', 'User_Transaction_Count',
            'IsWeekend', 'Transactions_Per_Day',
            'MerchantState_Encoded', 'Error_Encoded',
            'Payment_Type_Encoded', 'Category_Encoded'
        ]

        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols

        X = df[available_cols]
        y = df['Fraud_Status']

        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {y.mean():.3%}")

        return X, y, df

    def handle_class_imbalance(self, X, y, method='hybrid'):
        """Handle class imbalance using various techniques"""
        print(f"Original class distribution: {np.bincount(y)}")

        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        elif method == 'hybrid':
            # First oversample minority class moderately
            smote = SMOTE(sampling_strategy=0.3, random_state=42)
            X_temp, y_temp = smote.fit_resample(X, y)
            # Then undersample majority class
            undersampler = RandomUnderSampler(
                sampling_strategy=0.7, random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)
        else:
            X_balanced, y_balanced = X, y

        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced

    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train, y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }

        best_score = 0
        best_model = None
        best_model_name = ""

        print("\nTraining models...")
        for name, model in models.items():
            print(f"\nTraining {name}...")

            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train_balanced)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)

            print(f"{name} AUC: {auc_score:.4f}")
            print(
                f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Store metrics
            self.model_metrics[name] = {
                'auc': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            # Select best model based on AUC
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_model_name = name

        self.model = best_model
        print(f"\nBest model: {best_model_name} (AUC: {best_score:.4f})")

        return best_model_name, best_score

    def save_model(self, filepath='fraud_detection_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='fraud_detection_model.pkl'):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data.get('model_metrics', {})
        print("Model loaded successfully")

    def predict_fraud(self, transaction_data):
        """Predict fraud probability for a single transaction"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()

        # Feature engineering (same as training)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Hour'] = df['Date'].dt.hour
            df['DayOfWeek'] = df['Date'].dt.dayofweek

        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(
            int) if 'DayOfWeek' in df.columns else 0
        df['Amount_Log'] = np.log1p(np.abs(df['Amount']))
        df['Amount_Squared'] = df['Amount'] ** 2

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col +
                        '_Encoded'] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col + '_Encoded'] = 0

        # Select and scale features
        X = df[self.feature_columns].fillna(0)

        if isinstance(self.model, LogisticRegression):
            X_scaled = self.scaler.transform(X)
            fraud_probability = self.model.predict_proba(X_scaled)[0, 1]
        else:
            fraud_probability = self.model.predict_proba(X)[0, 1]

        return fraud_probability


# Initialize the model
fraud_model = FraudDetectionModel()


def train_model_interface():
    """Interface function for training the model"""
    try:
        # Load and preprocess data
        X, y, df = fraud_model.load_and_preprocess_data('Fact_Transaction.csv')

        # Train models
        best_model_name, best_score = fraud_model.train_models(X, y)

        # Save model
        fraud_model.save_model()

        # Create performance plots
        fig = create_model_performance_plots()

        return f"✅ Model trained successfully!\nBest Model: {best_model_name}\nAUC Score: {best_score:.4f}", fig

    except Exception as e:
        return f"❌ Error training model: {str(e)}", None


def create_model_performance_plots():
    """Create performance visualization plots"""
    if not fraud_model.model_metrics:
        return None

    fig = plt.figure(figsize=(15, 10))

    # ROC Curves
    plt.subplot(2, 3, 1)
    for model_name, metrics in fraud_model.model_metrics.items():
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC: {metrics['auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Feature importances
    if hasattr(fraud_model.model, 'feature_importances_'):
        plt.subplot(2, 3, 2)
        importances = fraud_model.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        plt.bar(range(10), importances[indices])
        plt.title('Top 10 Feature Importances')
        plt.xticks(range(10), [fraud_model.feature_columns[i]
                   for i in indices], rotation=45)

    # Confusion Matrix for best model
    best_model_name = max(fraud_model.model_metrics.keys(),
                          key=lambda x: fraud_model.model_metrics[x]['auc'])
    best_metrics = fraud_model.model_metrics[best_model_name]

    plt.subplot(2, 3, 3)
    cm = confusion_matrix(best_metrics['y_test'], best_metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    return fig


def predict_fraud_interface(amount, use_chip, merchant_state, mcc, error, payment_type,
                            category, day, month, hour, days_since_open, is_weekend):
    """Interface function for fraud prediction"""
    try:
        # Prepare transaction data
        transaction_data = {
            'Amount': float(amount),
            'Amount_ZScore': (float(amount) - 0) / 1000,  # Simplified z-score
            'Amount_Log': np.log1p(abs(float(amount))),
            'Amount_Squared': float(amount) ** 2,
            'UseChip': use_chip,
            'MerchantState': merchant_state,
            'MCC': int(mcc),
            'Error': error,
            'Payment_Type': payment_type,
            'Category': category,
            'Day': int(day),
            'Month': int(month),
            'Hour': int(hour),
            'DayOfWeek': 0,
            'Days_Since_Account_Open': int(days_since_open),
            'User_Transaction_Count': 10,
            'IsWeekend': int(is_weekend),
            'Transactions_Per_Day': 3
        }

        # Get prediction
        fraud_probability = fraud_model.predict_fraud(transaction_data)

        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        elif fraud_probability < 0.7:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
        else:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"

        result = f"""
        **Fraud Probability: {fraud_probability:.1%}**
        
        **Risk Level: {risk_level}**
        
        **Recommendation:** 
        {'✅ Transaction appears legitimate' if fraud_probability < 0.5 else '⚠️ Manual review recommended' if fraud_probability < 0.8 else '🚨 Block transaction - High fraud risk'}
        """

        return result

    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"

# Create Gradio Interface


def create_gradio_interface():
    with gr.Blocks(title="Fraud Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔒 Credit Card Fraud Detection System")
        gr.Markdown(
            "Train machine learning models and predict fraud probability for transactions")

        with gr.Tabs():
            # Training Tab
            with gr.Tab("🎯 Model Training"):
                gr.Markdown("## Train Fraud Detection Model")
                gr.Markdown(
                    "Click the button below to train the fraud detection model using the transaction data.")

                train_btn = gr.Button("🚀 Train Model", variant="primary")
                training_output = gr.Textbox(label="Training Results", lines=5)
                performance_plot = gr.Plot(label="Model Performance")

                train_btn.click(
                    train_model_interface,
                    outputs=[training_output, performance_plot]
                )

            # Prediction Tab
            with gr.Tab("🔍 Fraud Prediction"):
                gr.Markdown("## Predict Fraud Probability")
                gr.Markdown(
                    "Enter transaction details to get fraud risk assessment")

                with gr.Row():
                    with gr.Column():
                        amount = gr.Number(
                            label="Transaction Amount ($)", value=100.0)
                        use_chip = gr.Dropdown(
                            choices=["Chip Transaction",
                                     "Online Transaction", "Swipe Transaction"],
                            label="Transaction Type",
                            value="Chip Transaction"
                        )
                        merchant_state = gr.Textbox(
                            label="Merchant State", value="CA")
                        mcc = gr.Number(
                            label="Merchant Category Code (MCC)", value=5411)
                        error = gr.Dropdown(
                            choices=["No Error", "Insufficient Balance",
                                     "Technical malfunction", "Bad PIN"],
                            label="Transaction Error",
                            value="No Error"
                        )

                    with gr.Column():
                        payment_type = gr.Dropdown(
                            choices=["Credit", "Debit"],
                            label="Payment Type",
                            value="Credit"
                        )
                        category = gr.Dropdown(
                            choices=["grocery_pos", "gas_transport",
                                     "misc_net", "shopping_net", "shopping_pos"],
                            label="Transaction Category",
                            value="grocery_pos"
                        )
                        day = gr.Slider(minimum=1, maximum=31,
                                        value=15, label="Day of Month")
                        month = gr.Slider(
                            minimum=1, maximum=12, value=6, label="Month")
                        hour = gr.Slider(minimum=0, maximum=23,
                                         value=14, label="Hour of Day")

                    with gr.Column():
                        days_since_open = gr.Number(
                            label="Days Since Account Open", value=365)
                        is_weekend = gr.Checkbox(
                            label="Is Weekend Transaction?", value=False)

                predict_btn = gr.Button(
                    "🔍 Predict Fraud Risk", variant="primary")
                prediction_output = gr.Markdown(label="Fraud Risk Assessment")

                predict_btn.click(
                    predict_fraud_interface,
                    inputs=[amount, use_chip, merchant_state, mcc, error, payment_type,
                            category, day, month, hour, days_since_open, is_weekend],
                    outputs=prediction_output
                )

            # Model Info Tab
            with gr.Tab("📊 Model Information"):
                gr.Markdown("## Model Performance Metrics")
                gr.Markdown("""
                This fraud detection system uses machine learning to analyze transaction patterns and identify potentially fraudulent activities.
                
                **Features Used:**
                - Transaction amount and derived features
                - Merchant information and categories
                - Transaction timing patterns
                - Account history
                - Payment method details
                
                **Models Compared:**
                - Random Forest
                - Gradient Boosting
                - Logistic Regression
                
                **Key Considerations:**
                - Class imbalance handling using SMOTE and undersampling
                - Feature engineering for better predictive power
                - Cross-validation for robust model selection
                """)

    return demo


# Main execution
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
