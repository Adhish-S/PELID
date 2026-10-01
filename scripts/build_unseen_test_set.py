"""
Builds a strictly held-out, UNSEEN evaluation test set with 0% overlap with training_data.csv.
Covers real-world vernacular variations, slang, typos, and phrasing across English, Hinglish, and Manglish.
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "training_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "unseen_test_set.txt"

# Load all training texts (normalized) to enforce zero overlap
training_texts = set()
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            training_texts.add(row[1].strip().lower())

# Completely fresh, realistic test samples never seen in training
CANDIDATE_TEST_SET = [
    # ─── 1. order_status ───
    ("order_status", "Can someone give me an ETA on package tracking ID 88301?"),
    ("order_status", "Bhai mera parcel abhi kaunse delivery hub pe hai?"),
    ("order_status", "Ente shipment dispatch aayitt ethra divasam aayi, status check cheyyo?"),
    ("order_status", "Where did the courier person reach?"),
    ("order_status", "Package status track cheyyaanulla link onnu ayachu tharumo?"),
    ("order_status", "Mera consignment abhi tak out for delivery nahi dikha raha."),
    ("order_status", "Could you check where order number 994012 is right now?"),

    # ─── 2. order_delayed_or_missing ───
    ("order_delayed_or_missing", "It says delivered on the dashboard but my doorstep is empty!"),
    ("order_delayed_or_missing", "Delivery executive called once and marked as customer unavailable, totally false!"),
    ("order_delayed_or_missing", "Char din se package out for delivery par atka hua hai."),
    ("order_delayed_or_missing", "Item kittiyilla pakshe SMS-il delivered ennu vannu, enthu cheyyum?"),
    ("order_delayed_or_missing", "Parcel delay aayi, expected date kazhinjitt 2 days aayi."),
    ("order_delayed_or_missing", "Mera urgent packet abhi tak receive nahi hua, please expedite."),

    # ─── 3. wrong_or_defective_item ───
    ("wrong_or_defective_item", "The smartphone screen is completely shattered right out of the sealed box."),
    ("wrong_or_defective_item", "Maine black shoes mangwaye the, box ke andar white sneakers nikle."),
    ("wrong_or_defective_item", "Kittiya laptop on aakunnilla, power button work aavunnilla."),
    ("wrong_or_defective_item", "The parcel was crushed and liquid was leaking from the package."),
    ("wrong_or_defective_item", "Box mein original item gayab hai aur pathar bhara hua hai!"),
    ("wrong_or_defective_item", "Package pottiya avasthayilaanu kittiyathu, product damage aayi."),

    # ─── 4. refund_request [DESTRUCTIVE] ───
    ("refund_request", "Kindly credit the full purchase amount back into my bank account immediately."),
    ("refund_request", "Return pick-up ho chuka hai, mera 5000 rupaye ka refund kab aayega?"),
    ("refund_request", "Cancelled item-inte refund ethra divasathil bank-il kerum?"),
    ("refund_request", "I am fed up with the delays, give my payment back to my credit card!"),
    ("refund_request", "Mera UPI account mein refund transfer initiate hua ya nahi?"),
    ("refund_request", "Ente cash refund UPI-il thirike ayakkuka please."),

    # ─── 5. return_or_exchange ───
    ("return_or_exchange", "The shirt size is too tight around the chest, I need a replacement in XL."),
    ("return_or_exchange", "Mujhe ye watch pasand nahi aayi, dusre model ke saath replace karna hai."),
    ("return_or_exchange", "Ith mattonnu aayi maattaan pattumo? Fitting shariyalla."),
    ("return_or_exchange", "Please schedule a courier guy to collect this item for return."),
    ("return_or_exchange", "Exchange policy ethra divasam vare undu?"),

    # ─── 6. cancel_order [DESTRUCTIVE] ───
    ("cancel_order", "Accidentally placed the order with the wrong quantity, stop the shipment!"),
    ("cancel_order", "Mujhe ye order abhi nahi chahiye, turant cancel kardo."),
    ("cancel_order", "Ente order cancel cheyyanulla request accept cheyyumo?"),
    ("cancel_order", "Halt the warehouse dispatch and void order #10294."),
    ("cancel_order", "Please abort my booking before the delivery boy departs."),

    # ─── 7. cancel_subscription [DESTRUCTIVE] ───
    ("cancel_subscription", "Discontinue my annual Prime membership and turn off auto-debit."),
    ("cancel_subscription", "Mujhe monthly subscription end karna hai, agle mahine paise mat kaatna."),
    ("cancel_subscription", "Auto-renewal off cheyyan vendi settings evideyaanu?"),
    ("cancel_subscription", "Please terminate my recurring subscription plan effective today."),

    # ─── 8. billing_dispute [DESTRUCTIVE] ───
    ("billing_dispute", "I was charged twice on my credit card statement for a single checkout!"),
    ("billing_dispute", "Mere bank account se bina OTP ke do baar amount deduct ho gaya."),
    ("billing_dispute", "Bill-il extra amount charge cheythittundu, athu enthaanu?"),
    ("billing_dispute", "There is an unauthorized $89 charge on my invoice, reverse this fraud."),

    # ─── 9. payment_failure ───
    ("payment_failure", "My Google Pay failed at the gateway but money was debited from my bank."),
    ("payment_failure", "Transaction fail ho gaya par order confirmed nahi dikha raha."),
    ("payment_failure", "UPI payment timeout aayi, balance pakshe account-il ninnu poyi."),
    ("payment_failure", "The payment window crashed right after entering the CVV."),

    # ─── 10. invoice_or_receipt ───
    ("invoice_or_receipt", "I require a stamped tax invoice with our corporate GST number for filing."),
    ("invoice_or_receipt", "Mujhe company reimbursement ke liye formal bill copy chahiye."),
    ("invoice_or_receipt", "GST bill download cheyyaanulla option app-il evideyaanu?"),
    ("invoice_or_receipt", "Could you email the official purchase receipt for last week's order?"),

    # ─── 11. login_or_password_issue ───
    ("login_or_password_issue", "The 6-digit verification OTP is not arriving on my mobile phone."),
    ("login_or_password_issue", "Account lock ho gaya hai aur password reset email nahi aa raha."),
    ("login_or_password_issue", "Login cheyyaan pattunnilla, invalid credentials ennu kaanikunnu."),
    ("login_or_password_issue", "Forgot my two-factor authenticator app backup codes."),

    # ─── 12. delete_account [DESTRUCTIVE] ───
    ("delete_account", "Under GDPR right to erasure, purge all my personal profile data and delete my account."),
    ("delete_account", "Mujhe apna account permanently band karna hai aur saara data delete karo."),
    ("delete_account", "Ente account full aayi delete cheyyanam, profile close cheyyo."),
    ("delete_account", "Wipe out all my saved addresses, credit cards, and close this profile forever."),

    # ─── 13. update_account_details ───
    ("update_account_details", "How do I update my registered mobile number after switching SIM cards?"),
    ("update_account_details", "Mera default billing address aur flat number change karna hai."),
    ("update_account_details", "Profile-il puthiya email ID engane add cheyyam?"),
    ("update_account_details", "Can I change my recipient contact name on my primary account?"),

    # ─── 14. shipping_address_change [DESTRUCTIVE] ───
    ("shipping_address_change", "Can you reroute the package to my office address since I am out of station?"),
    ("shipping_address_change", "Dispatch hone se pehle delivery location Dusre pincode pe shift kardo."),
    ("shipping_address_change", "Order cheytha item vere sthalathekku redirect cheyyan pattumo?"),
    ("shipping_address_change", "Update the delivery street to Building 4B instead of Building 2A."),

    # ─── 15. product_inquiry_or_specs ───
    ("product_inquiry_or_specs", "Does this wireless headphone support active noise cancellation and multipoint pairing?"),
    ("product_inquiry_or_specs", "Is laptop mein RAM upgrade karne ke liye extra slot hai kya?"),
    ("product_inquiry_or_specs", "Ee phone-il water resistance rating undo? Battery capacity ethra aanu?"),
    ("product_inquiry_or_specs", "What is the warranty period for this electric kettle?"),

    # ─── 16. how_to_question ───
    ("how_to_question", "Step by step instructions on how to redeem reward points at checkout?"),
    ("how_to_question", "Wishlist se cart mein items kaise move karte hain?"),
    ("how_to_question", "Gift coupon engane apply cheyyam payment page-il?"),
    ("how_to_question", "How do I set up recurring monthly deliveries for groceries?"),

    # ─── 17. pricing_or_plan_question ───
    ("pricing_or_plan_question", "Is there any seasonal festival discount available on the premium plan?"),
    ("pricing_or_plan_question", "Delivery charge free aano 500 roopaykku mukalil purchase cheythaal?"),
    ("pricing_or_plan_question", "Do you offer student discounts or corporate bulk pricing?"),

    # ─── 18. complaint_or_escalate_to_human ───
    ("complaint_or_escalate_to_human", "Stop giving me robotic scripted replies and connect me to a human manager!"),
    ("complaint_or_escalate_to_human", "Mujhe customer care ke senior supervisor se turant baat karni hai."),
    ("complaint_or_escalate_to_human", "Oru human executive-nod samsaarikkanam, bot reply venda."),
    ("complaint_or_escalate_to_human", "This is the worst customer service ever, I am going to file a consumer forum complaint."),

    # ─── 19. feature_request_or_feedback ───
    ("feature_request_or_feedback", "It would be wonderful if your app supported dark mode on iOS."),
    ("feature_request_or_feedback", "App ka checkout interface kaafi fast aur smooth ho gaya hai, great work!"),
    ("feature_request_or_feedback", "Malayalam language support app-il koodi koduthaal nallathaayirikkum."),
    ("feature_request_or_feedback", "Please consider adding Apple Pay and crypto checkout options."),

    # ─── 20. general_chitchat_or_faq ───
    ("general_chitchat_or_faq", "Good morning team! Hope you all are having a productive day."),
    ("general_chitchat_or_faq", "Namaste bhai, kaise ho?"),
    ("general_chitchat_or_faq", "Sugham aano? Nice talking to you."),
    ("general_chitchat_or_faq", "Who founded this e-commerce platform and where is your headquarters?"),

    # ─── 21. other_unclear ───
    ("other_unclear", "Wait a second, let me check something else first..."),
    ("other_unclear", "Arre kya chal raha hai yahan par kuch samajh nahi aa raha"),
    ("other_unclear", "Enthokkeyo paranj poyi, pinne nokkaam"),
    ("other_unclear", "random gibberish xyz123 foo bar"),
    ("other_unclear", "hmm ok fine whatever"),
]

# Verify strictly zero overlap with training data
strictly_unseen = []
skipped = 0
for label, query in CANDIDATE_TEST_SET:
    norm = query.strip().lower()
    if norm in training_texts:
        print(f"[WARNING] Overlap detected and filtered: {query}")
        skipped += 1
    else:
        strictly_unseen.append((label, query))

print(f"Total candidate samples: {len(CANDIDATE_TEST_SET)}")
print(f"Overlapping samples filtered: {skipped}")
print(f"Guaranteed Unseen Test Samples: {len(strictly_unseen)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("# Strictly Held-Out Unseen Test Set (Guaranteed 0% Training Overlap)\n")
    f.write("# Format: label | query\n\n")
    for label, query in strictly_unseen:
        f.write(f"{label} | {query}\n")

print(f"[OK] Saved to: {OUTPUT_FILE}")
