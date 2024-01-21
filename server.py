import streamlit as st 
import pickle

model = pickle.load(open('model_logistic.pkl', 'rb'))

def predict_pro(list1):
    
    print("inside predict function")
    prediction=model.predict(list1)
   
    return prediction

def main():

    pred_arr = []

    PreferredLoginDevice_dict = {"Mobile Phone":(1,0),"Phone":(0,1),"Computer":(0,0)}

    PreferredPaymentMode_dict = {"COD":(1,0,0,0,0,0),"Cash on Delivery":(0,1,0,0,0,0),"Credit Card":(0,0,1,0,0,0),
                                 "Debit Card":(0,0,0,1,0,0),"E Wallet":(0,0,0,0,1,0),"UPI":(0,0,0,0,0,1),
                                 "Debit Card":(0,0,0,0,0,0)}
    
    gender_Male_dict = {"Male":1,"Female":0}

    PreferedOrderCat_dict = {"Grocery":(1,0,0,0,0),"Laptop & Accessory":(0,1,0,0,0),"Mobile":(0,0,1,0,0),
                                 "Mobile Phone":(0,0,0,1,0),"Others":(0,0,0,0,1),"Fashion":(0,0,0,0,0)}
    
    MaritalStatus_dict = {"Married":(1,0),"Single":(0,1),"Divorced":(0,0)}

    yes_no_dict = {"Yes":1,"No":0}
    
    st.title("E Commerce Churn Prediction")
    html_temp = """
    <div style="background-color:#b3db86;padding:8px">
    <h2 style="color:black;text-align:center;">Predicting Customer Churn</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    tenure = st.slider("Enter tenure (months)", min_value=0, max_value=500)
    pred_arr.extend([tenure])
    # print(pred_arr)

    CityTier = st.slider("CityTier", min_value=0, max_value=3)
    pred_arr.extend([CityTier])
    # print(pred_arr)

    WarehouseToHome = st.slider("WarehouseToHome", min_value=0, max_value=15000)
    pred_arr.extend([WarehouseToHome])
    # print(pred_arr)

    HourSpendOnApp = st.slider("HourSpendOnApp", min_value=0, max_value=15000)
    pred_arr.extend([HourSpendOnApp])

    NumberOfDeviceRegistered = st.slider("NumberOfDeviceRegistered", min_value=0, max_value=5)
    pred_arr.extend([NumberOfDeviceRegistered])

    SatisfactionScore = st.slider("SatisfactionScore", min_value=0, max_value=5)
    pred_arr.extend([SatisfactionScore])

    NumberOfAddress = st.slider("NumberOfAddress", min_value=0, max_value=5)
    pred_arr.extend([NumberOfAddress])

    OrderAmountHikeFromlastYear = st.slider("OrderAmountHikeFromlastYear", min_value=0, max_value=5)
    pred_arr.extend([NumberOfAddress])

    CouponUsed = st.slider("CouponUsed", min_value=0, max_value=6)
    pred_arr.extend([CouponUsed])

    OrderCount = st.slider("OrderCount", min_value=0, max_value=15)
    pred_arr.extend([OrderCount])

    DaySinceLastOrder = st.slider("DaySinceLastOrder", min_value=0, max_value=15)
    pred_arr.extend([DaySinceLastOrder])

    CashbackAmount = st.slider("CashbackAmount", min_value=0, max_value=15)
    pred_arr.extend([CashbackAmount])

    PreferredLoginDevice_Mobile_Phone = st.selectbox("PreferredLoginDevice",("Mobile Phone", "Phone", "Computer"))
    pred_arr.extend(PreferredLoginDevice_dict[PreferredLoginDevice_Mobile_Phone])

    PreferredPaymentMode_COD = st.selectbox("PreferredPaymentMode",("COD", "Cash on Delivery", "Credit Card", "Debit Card",
                                                                    "E Wallet","UPI","CC"))
    pred_arr.extend(PreferredPaymentMode_dict[PreferredPaymentMode_COD])

    gender_Male = st.selectbox("Gender",("Male", "Female"))
    pred_arr.extend([gender_Male_dict[gender_Male]])
    # print(pred_arr)

    PreferedOrderCat_Grocery = st.selectbox("PreferedOrderCat",("Grocery", "Laptop & Accesory", "Mobile", "Mobile Phone", "Others", "Fashion"))
    pred_arr.extend(PreferedOrderCat_dict[PreferedOrderCat_Grocery])

    MaritalStatus_Married = st.selectbox("MaritalStatus",("Married", "Single", "Divorced"))
    pred_arr.extend(MaritalStatus_dict[MaritalStatus_Married])

    Complain_Yes = st.selectbox("Complain", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[Complain_Yes]])
    
    result=""
    if st.button("Predict"):
        
        print(len(pred_arr))
        # pred_arr = [[1,89.35,89.35,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0]]
        result = predict_pro([pred_arr])

        print("Printing result")
        print(result)
        
        if result[0] == 0:
                st.success('The customer will not churn!', icon="✅")
        else:
             st.error("the custumoer is likely to churn!")
    if st.button("About"):
        st.text("V:2, Model used: Logitic Regression")

main()
