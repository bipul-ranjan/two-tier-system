"""
Generates synthetic customer-support data in the Bitext retail-banking
format (columns: instruction, category, intent, response), as several CSV
files, each a random mix of four scenario groups:

  payments_normal     -> Bitext categories CARD, TRANSFER, ATM, FEES
  retail_normal       -> Bitext categories ACCOUNT, LOAN, PASSWORD, CONTACT, FIND
  payments_exception  -> category PAYMENT_EXCEPTION (fraud, duplicate charge,
                         chargeback, stuck/wrong transfers, compliance holds ...)
  retail_exception    -> category RETAIL_EXCEPTION (frozen account, bereavement,
                         hardship, identity theft, complaints, vulnerable customers ...)

"Without formatting" is implemented as: plain ASCII text, single line per
cell, no {{placeholders}}, no markdown, no training-template markup.

All bank policies, time frames and steps in the responses are SYNTHETIC
and generic -- they are not the policies of any real bank.

Reproducible: same SEED -> same files. Run from the project root:
    python scripts/generate_synthetic_bitext.py
"""
import csv
import os
import random
from string import Formatter

SEED = 20260928
N_FILES = 10
ROWS_PER_FILE = 1600          # requirement: at least 1500 per file
OUT_DIR = "data/synthetic"
COLUMNS = ["instruction", "category", "intent", "response"]
GROUPS = ["payments_normal", "retail_normal", "payments_exception", "retail_exception"]

# ----------------------------------------------------------------- slot pools
MERCHANTS = ["Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Netflix", "Spotify", "Apple",
             "Google Play", "BigBasket", "MakeMyTrip", "Booking.com", "Airbnb", "Myntra", "IKEA",
             "Starbucks", "Decathlon", "Reliance Digital", "Expedia", "Steam", "Nykaa", "Ola", "PayPal"]
CARD_TYPES = ["debit card", "credit card", "prepaid card", "travel card", "platinum credit card"]
ACCOUNT_TYPES = ["savings account", "current account", "salary account", "joint account",
                 "student account", "business account"]
LOAN_TYPES = ["personal loan", "home loan", "car loan", "student loan", "business loan"]
CITIES = ["Paris", "Dubai", "Singapore", "London", "Bangkok", "New York", "Sydney", "Tokyo", "Toronto",
          "Rome", "Berlin", "Istanbul", "Mumbai", "Pune", "Delhi", "Bangalore", "Chennai", "Hyderabad",
          "Dublin", "Amsterdam"]
LANDMARKS = ["the airport", "the central railway station", "the main market", "my office", "the city mall",
             "the university campus", "the bus terminal", "the old town square"]
WHEN = ["yesterday", "this morning", "last Friday", "two days ago", "last week", "earlier today",
        "last night", "last weekend", "three days ago", "last month"]


def fmt_amount(rng, low, high):
    n = rng.randint(low, high)
    if n >= 1000:
        n = round(n, -1)
    s = f"{n:,}"
    style = rng.randint(0, 5)
    if style == 0:
        return f"${s}"
    if style == 1:
        return f"GBP {s}"
    if style == 2:
        return f"EUR {s}"
    if style == 3:
        return f"INR {s}"
    if style == 4:
        return f"{s} dollars"
    return f"{s} USD"


def make_slots(rng):
    return {
        "merchant": rng.choice(MERCHANTS),
        "amount": fmt_amount(rng, 15, 3000),
        "big_amount": fmt_amount(rng, 5000, 60000),
        "loan_amount": fmt_amount(rng, 2000, 500000),
        "card_type": rng.choice(CARD_TYPES),
        "last4": f"{rng.randint(0, 9999):04d}",
        "when": rng.choice(WHEN),
        "city": rng.choice(CITIES),
        "days": rng.choice([2, 3, 4, 5, 6, 7, 10]),
        "weeks": rng.choice([2, 3, 4, 6, 8]),
        "tenure": rng.choice([12, 24, 36, 48, 60, 84]),
        "account_type": rng.choice(ACCOUNT_TYPES),
        "loan_type": rng.choice(LOAN_TYPES),
        "landmark": rng.choice(LANDMARKS),
    }


# --------------------------------------------------------- style / phrasing
PREFIXES = ["", "", "", "Hi, ", "Hello, ", "Hey, ", "Good morning, ", "Good evening, ", "Excuse me, ",
            "Please help: ", "Quick question: ", "I need help. ", "Urgent: ", "Hi team, ", "Hello there, "]
SUFFIXES = ["", "", "", " Thanks.", " Thank you.", " Please help.", " Can you help me with this?",
            " Any idea what to do?", " I need this sorted today.", " Please advise.",
            " Appreciate the help.", " Not sure what to do."]
OPEN_HELP = ["Certainly, I can help with that.", "Of course, happy to help.", "Sure, here is how this works.",
             "Thanks for reaching out, I can guide you through this.", "Absolutely, let me walk you through it."]
OPEN_EMPATHY = ["I am sorry to hear about this and I want to help sort it out.",
                "I understand how stressful this must be, and I am here to help.",
                "I apologise for the trouble this has caused, let me help.",
                "That sounds frustrating, and I will do my best to help you resolve it."]
CLOSE_GEN = ["If you need any further help, just let me know.", "Please reach out again if anything is unclear.",
             "Feel free to ask if you have any other questions.", "I am here if you need anything else.",
             "Let me know if there is anything more I can do for you."]
CLOSE_ESC = ["If you would like, I can connect you with a human agent right now.",
             "You can ask for a human agent at any time and I will arrange it.",
             "Please keep your reference number safe, and reply here if you need anything else.",
             "We will keep you updated by message until this is resolved.",
             "If anything changes in the meantime, please let us know straight away."]


def lower_first(text):
    if len(text) > 1 and (text[1].isupper() or text[:2] in ("I ", "I'")):
        return text
    return text[0].lower() + text[1:]


def style(rng, text):
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    if prefix and (prefix.endswith(", ") or prefix.endswith(": ")):
        text = lower_first(text)
    if suffix and text[-1] not in ".?!":
        text += "."
    s = prefix + text + suffix
    if rng.random() < 0.18:
        s = s.lower()
    if rng.random() < 0.25:
        s = s.rstrip(".?! ")
    if rng.random() < 0.25:
        for a, b in [("please", "pls"), ("Please", "Pls"), (" you ", " u "), ("cannot", "cant"),
                     ("do not", "dont"), ("I am", "im"), ("Thanks", "thx")]:
            if a in s and rng.random() < 0.5:
                s = s.replace(a, b)
    return s


def build_response(rng, tone, body):
    parts = []
    if tone == "help" and rng.random() < 0.5:
        parts.append(rng.choice(OPEN_HELP))
    elif tone == "empathy" and rng.random() < 0.6:
        parts.append(rng.choice(OPEN_EMPATHY))
    parts.append(body)
    if tone in ("help", "empathy") and rng.random() < 0.7:
        parts.append(rng.choice(CLOSE_GEN))
    elif tone == "escalate" and rng.random() < 0.85:
        parts.append(rng.choice(CLOSE_ESC))
    elif tone == "none" and rng.random() < 0.5:
        parts.append(rng.choice(CLOSE_GEN))
    return " ".join(parts)


def fields(template):
    return {f for _, f, _, _ in Formatter().parse(template) if f}


# ------------------------------------------------------------------- intents
# (category, intent, tone, [instruction templates], [response bodies])
# Convention: the LAST response body of every intent uses no {slots}, so a
# response never mentions a detail the customer did not state.
INTENTS = {
    "payments_normal": [
        ("CARD", "activate_card", "help",
         ["How do I activate my new {card_type}?",
          "I just received my {card_type} in the mail, how can I start using it?",
          "My {card_type} ending {last4} arrived today. What are the steps to activate it?",
          "Can you help me activate my card? It has not been used yet.",
          "Where do I activate the {card_type} I got {when}?",
          "Need to switch on my new card, what should I do?"],
         ["First, open the mobile app or net banking and go to the Cards section. Then select your {card_type} and choose Activate Card. You will receive a one-time code on your registered mobile number, so enter it to confirm. Once that is done, the card is active and ready to use straight away.",
          "You can activate your card in two easy ways. In the mobile app, go to Cards, pick your card, and tap Activate. If you prefer, call the phone banking number printed on the back of the card and follow the voice prompts. After verification your card will be active within a few minutes."]),
        ("CARD", "activate_card_international_usage", "help",
         ["How can I enable my {card_type} for use abroad? I am travelling to {city}.",
          "I am going to {city} soon. Can you turn on international usage for my card?",
          "My {card_type} does not work outside the country, how do I switch on overseas payments?",
          "Please enable international transactions on my card ending {last4}.",
          "What is the process to allow foreign transactions on my card before my trip?"],
         ["International usage can be switched on from the mobile app. Go to Cards, select your {card_type}, open Card Controls and turn on International Transactions. You can also set the dates of your trip and a spending limit for extra safety. Changes usually take effect within a few minutes, so please do it before you leave.",
          "To use your card overseas, open Card Controls in the app and enable International Usage. It is a good idea to set a travel period and a daily limit. If you cannot use the app, phone banking can turn it on for you after verifying your identity."]),
        ("CARD", "block_card", "empathy",
         ["I lost my {card_type} and need to block it right now.",
          "Please block my card ending {last4}, I think it was stolen.",
          "How do I freeze my card temporarily? I cannot find it.",
          "My card went missing {when}, block it asap.",
          "Someone may have my card details, how can I block the card immediately?"],
         ["You can block the card immediately in the mobile app. Go to Cards, choose the card and tap Block Card, then pick Lost or Stolen as the reason. The card stops working straight away and any new transactions are declined. We will then help you order a replacement, which usually arrives in a few working days.",
          "To protect your money, block the card first. Use the app under Cards and Block, or call the 24 hour phone banking line and ask for an immediate block. Once blocked, please review your recent transactions and tell us about any you do not recognise so we can look into them."]),
        ("CARD", "cancel_card", "help",
         ["I want to cancel my {card_type} permanently.",
          "How can I close my credit card? I no longer use it.",
          "Please cancel the card ending {last4}.",
          "What happens if I cancel my {card_type}, and how do I do it?",
          "I would like to stop my card, it has an annual fee I do not want to pay."],
         ["To cancel a card, go to Cards in the mobile app, select your {card_type} and choose Cancel Card, or contact customer service. Before we close it, please clear any outstanding balance and cancel recurring payments linked to the card. Once cancelled, the card cannot be used again, and you can request a new one later if you change your mind.",
          "I can guide you through cancelling your card. First check that there are no pending transactions or unpaid dues. Then submit a cancellation request through the app or by phone banking. You will get a confirmation message once the card is closed."]),
        ("CARD", "check_card_annual_fee", "help",
         ["What is the annual fee for my {card_type}?",
          "How much do I pay every year to keep my card?",
          "Could you tell me the yearly charges on the card ending {last4}?",
          "Is there any joining or annual fee on the {card_type}?",
          "When is my card annual fee charged and can it be waived?"],
         ["The annual fee depends on the type of card you hold. You can see the exact amount for your {card_type} in the app under Cards and Card Details, and it is also listed in your welcome kit. The fee is normally charged once a year on the card anniversary date. Some cards offer a waiver if you reach a yearly spending target.",
          "Annual fees vary by card, and many cards waive the fee after a certain level of yearly spending. Please check Fees and Charges for your card in the app to see your amount and the anniversary date. If you would like to ask about a waiver, customer service can review your eligibility."]),
        ("CARD", "check_current_balance_on_card", "help",
         ["What is the available balance on my {card_type}?",
          "How can I check how much credit I have left on my card?",
          "Please tell me the current balance on the card ending {last4}.",
          "How do I see my card balance and available limit?",
          "I want to know my outstanding amount on the credit card."],
         ["You can see your card balance in the mobile app under Cards. Select your {card_type} to view the available limit, the outstanding amount and the payment due date. You can also get a quick balance by sending a text message from your registered number or by calling phone banking.",
          "To check your balance, log in to net banking or the app and open Cards. The card summary shows your current balance, available credit and the last statement amount. Balances update in near real time, but very recent transactions can take a short while to appear."]),
        ("TRANSFER", "cancel_transfer", "help",
         ["I sent {amount} to the wrong person, can I cancel the transfer?",
          "How do I stop a transfer I made {when}?",
          "Please cancel my pending transfer of {amount}.",
          "I want to reverse the payment I just made, is that possible?",
          "Can I cancel a scheduled transfer for next week?"],
         ["If the transfer of {amount} is still pending or scheduled, you can cancel it in the app. Go to Transfers, open Pending or Scheduled, select the transfer and tap Cancel. If it has already been completed, it cannot be cancelled directly, but you can raise a request and we will contact the receiving bank to ask for the money back. Success depends on the receiving bank and the account holder agreeing to return it.",
          "Please act quickly. For transfers that have not been processed yet, use Cancel next to the transfer in your pending list. For completed transfers, raise a recall request from the same screen or through customer service, and keep the transaction reference handy. We will update you as soon as we hear back from the other bank."]),
        ("TRANSFER", "make_transfer", "help",
         ["How do I send {amount} to a friend's account?",
          "I need to transfer money to another bank, what are the steps?",
          "Please help me make a transfer of {amount} today.",
          "What is the fastest way to send money to someone?",
          "How can I add a new payee and transfer funds?"],
         ["To make a transfer, log in to the app or net banking and go to Transfers. Add the payee by entering the name, account number and bank code, then confirm the payee with a one-time code. Enter the amount of {amount}, choose the payment method, review the details and confirm. You will get a confirmation message with a reference number.",
          "You can send money in a few steps. Open Transfers, choose Send Money and select an existing payee or add a new one. Type in the amount, check the details carefully because transfers to the wrong account can be hard to reverse, and approve with your security code. Instant methods arrive within minutes, while standard transfers can take up to one working day."]),
        ("ATM", "dispute_ATM_withdrawal", "none",
         ["The ATM did not give me the full cash but the money was deducted, I withdrew {amount}.",
          "I want to dispute an ATM withdrawal of {amount} from {when}.",
          "Cash was not dispensed at the ATM yet my account was debited.",
          "There is an ATM transaction I do not recognise for {amount}, how do I raise a dispute?",
          "The ATM gave me less money than what was debited from my account."],
         ["I will help you raise a dispute for the ATM withdrawal of {amount}. Please go to Disputes in the app, select the transaction and choose ATM Cash Issue, then give the date, time and ATM location if you have them. We will investigate with the ATM operator and most cases are reviewed within a few working days. If the ATM did not dispense the cash, the amount is usually credited back to your account.",
          "Sorry about the trouble with the ATM. Please keep the receipt if you have one and raise a dispute through the app or by calling customer service. We will check the ATM records for the withdrawal and update you by message. If the money is confirmed as not dispensed, it will be returned to your account."]),
        ("ATM", "recover_swallowed_card", "none",
         ["The ATM swallowed my card, what should I do?",
          "My card got stuck in the machine {when}.",
          "The ATM kept my {card_type} and I cannot get it back.",
          "Help, the cash machine did not return my card.",
          "ATM retained my card ending {last4}, how do I recover it?"],
         ["Please do not worry, this happens sometimes. First, block the card in the app as a precaution until you have it back. Then contact the branch that owns the ATM or call customer service, and we will check if the card was retained by the machine. If it can be recovered you may collect it with valid photo ID, otherwise we will issue a replacement card to you.",
          "I am sorry that the ATM kept your card. For your safety, I suggest blocking the card right away in the app. We will then arrange either collection from the ATM branch with identification or a replacement card sent to your address, usually within a few working days."]),
        ("FEES", "check_fees", "help",
         ["What fees do you charge on my account?",
          "How much is the fee for an international transfer?",
          "Can you list all the charges for using the ATM abroad?",
          "Is there a fee if my balance drops below the minimum?",
          "Tell me about the monthly maintenance fees on my {account_type}.",
          "Why was I charged a fee of {amount} this month?"],
         ["Fees depend on the type of account or service. You can find the full list in the app under Help and then Fees and Charges, including maintenance, ATM, transfer and card fees. If you tell me the specific service you are asking about, I can explain how that fee is calculated and when it applies.",
          "I can help you understand the charges. Common fees include monthly maintenance, foreign transaction fees, ATM use outside our network and penalties for low balance. Your latest statement shows every fee that has been applied, and customer service can review any fee you believe was charged in error."]),
    ],
    "retail_normal": [
        ("ACCOUNT", "check_recent_transactions", "help",
         ["Can I see my last transactions?",
          "How do I view what I spent this week?",
          "Show me my recent activity on the {account_type}.",
          "Where can I download my statement for the last {days} days?",
          "I want to check if the payment I made {when} has gone through."],
         ["You can see recent transactions in the app or net banking. Open your {account_type} and tap Transactions to view the latest activity, and use the filters to search by date or amount. Statements for earlier periods can be downloaded as a PDF from the Statements section.",
          "To review your recent activity, log in and choose the account, then select Recent Transactions. Pending items are shown separately from completed ones. If something looks unfamiliar, tap it to see the details and report it directly from that screen."]),
        ("ACCOUNT", "close_account", "help",
         ["I want to close my {account_type}.",
          "How do I shut down my account permanently?",
          "Please help me close the account, I am moving abroad.",
          "What do I need to do before closing my {account_type}?",
          "Cancel my savings account and send me the balance."],
         ["I can explain how to close your {account_type}. Please first clear any dues, cancel standing instructions and move the remaining balance to another account. Then submit a closure request through customer service or at a branch with valid identification. Once processed, you will receive written confirmation and the final statement.",
          "To close the account, make sure there are no pending transactions, linked cards or loans. Request closure through the app under Account Services or visit a branch. We will transfer any remaining balance to the account you nominate and confirm once the account is closed."]),
        ("ACCOUNT", "create_account", "help",
         ["How do I open a {account_type}?",
          "I want to open a new account, what documents are required?",
          "Can I open an account online without visiting the branch?",
          "I am a student and would like to start a savings account.",
          "Please guide me through creating a new account."],
         ["Opening a {account_type} is simple. You can apply online or at a branch by providing a valid photo ID, proof of address and a recent photograph. After we verify your details, the account is created and you will receive your account number and access to the app. Some accounts require a small opening deposit.",
          "To start, choose the type of account that suits you, then fill in the application form online. You will need identity and address documents, and you may complete a short video verification. Once approved, your account details arrive by message and email, and your debit card is sent to your address."]),
        ("LOAN", "apply_for_loan", "help",
         ["I want to apply for a {loan_type} of {loan_amount}.",
          "How do I apply for a personal loan?",
          "What are the eligibility criteria for a {loan_type}?",
          "Can I get a loan for {loan_amount} over {tenure} months?",
          "Please explain how to apply for a loan online."],
         ["You can apply for a {loan_type} online or at a branch. Please keep your identity proof, address proof, recent salary slips or income proof, and bank statements ready. After you submit the application, we check your eligibility and credit history and share an offer with the interest rate and repayment plan. Once you accept, the money is usually credited within a few working days.",
          "To apply for a loan, go to Loans in the app and select the type of loan you need. Enter the amount and tenure, upload the required documents and submit. Your eligibility is based on income, existing commitments and credit score, and you will see a decision and an offer once the review is complete."]),
        ("LOAN", "apply_for_mortgage", "help",
         ["I would like to apply for a home loan.",
          "What is the process to get a mortgage for a house worth {loan_amount}?",
          "How much deposit do I need for a mortgage?",
          "Can you explain the steps for a home loan application?",
          "We are buying our first home and need a mortgage, where do we start?"],
         ["For a mortgage, the first step is to check how much you can borrow using the mortgage calculator in the app. Then submit an application with your identity, income and property details, and we will arrange a valuation of the property. After approval you receive a formal offer, and the funds are released to the seller once the legal formalities are complete. A deposit is usually needed, and the amount depends on the property value and your profile.",
          "Applying for a home loan involves a few stages: an initial eligibility check, document submission, property valuation, formal approval and legal completion. A mortgage adviser can guide you at each stage. You can begin by booking a call from the Loans section of the app."]),
        ("LOAN", "cancel_loan", "help",
         ["I want to cancel my {loan_type} application.",
          "How do I withdraw my loan request?",
          "Please cancel the loan I applied for {when}.",
          "Can I cancel my loan after it was approved but before receiving the money?"],
         ["If your {loan_type} has not been disbursed yet, you can cancel the request through customer service or from Loans in the app by selecting your application and choosing Withdraw. If the money has already been credited, you will need to repay it, and we can share the foreclosure amount with you. Please note that some fees may apply after disbursal.",
          "To cancel a loan application, tell us which application you mean and we will stop the process. Before disbursal there is normally no cost. After disbursal, the loan must be closed by repaying the outstanding amount, and we will give you the exact figure and any applicable charges."]),
        ("LOAN", "cancel_mortgage", "help",
         ["I want to cancel my mortgage application.",
          "How can I withdraw my home loan request?",
          "We decided not to buy the house, how do I cancel the mortgage?",
          "What happens to my mortgage application if I cancel it now?"],
         ["I can help you withdraw the mortgage application. Please contact our mortgage team through the app or at a branch and confirm the application reference. If the valuation or legal work has already started, some third party fees might still apply. Once cancelled, you will receive written confirmation.",
          "To cancel a mortgage application, send a withdrawal request with your application reference. Any costs already incurred for valuation or legal checks may not be refundable. We will confirm the cancellation in writing and close the file."]),
        ("LOAN", "check_loan_payments", "help",
         ["When is my next {loan_type} instalment due?",
          "How much do I still owe on my loan?",
          "Can I see my loan repayment schedule?",
          "I want to know how many payments are left on my loan.",
          "What is the outstanding balance on my {loan_type}?"],
         ["You can see your repayment details in the app under Loans. Select your {loan_type} to view the instalment amount, the next due date, the number of payments remaining and the total outstanding balance. The full repayment schedule can also be downloaded as a document.",
          "To check your loan payments, log in and open Loans, then choose the loan. You will see what you have paid so far, what is left, and the next due date. If you would like to pay extra to reduce the balance, there is an option to make a part payment on the same screen."]),
        ("LOAN", "check_mortgage_payments", "help",
         ["How much is my monthly mortgage payment?",
          "When is my next mortgage payment due?",
          "What is the remaining balance on my mortgage?",
          "Can I see how much interest I have paid on my home loan?"],
         ["In the app, open Loans and select your mortgage to see your monthly payment, the next due date, the remaining balance and the split between interest and principal. Statements and a yearly interest certificate can also be downloaded from the documents section.",
          "Your mortgage details are available under Loans. There you can view the payment amount, upcoming due dates, the balance still owed and a summary of interest paid. If your payment changes because of a rate change, we will notify you in advance."]),
        ("PASSWORD", "get_password", "none",
         ["I forgot my password, how do I get it back?",
          "I cannot log in to net banking, I do not remember my password.",
          "Please help me recover my login details.",
          "How can I reset my forgotten mobile banking password?"],
         ["I am sorry you are locked out. On the login page, tap Forgot Password and enter your customer ID. We will send a one-time code to your registered mobile number, and after you enter it you can choose a new password. For security reasons we can never tell you your old password.",
          "You can recover access in a few steps. Select Forgot Password on the login screen, verify your identity with the one-time code sent to your registered number, and set a new password. If you no longer have access to that number, please visit a branch with your identification to update it."]),
        ("PASSWORD", "set_up_password", "help",
         ["How do I set up a password for net banking?",
          "I am a new customer and want to create my login.",
          "What are the rules for choosing a password?",
          "How can I change my password to something stronger?"],
         ["To set up or change your password, log in and go to Settings and then Security. Choose Change Password, enter your current password, and type the new one twice. A strong password uses at least eight characters with a mix of letters, numbers and symbols, and should not include your name or birth date.",
          "New customers can create a password using the First Time Login option. Enter your customer ID, verify with a one-time code, then choose a password that meets the rules shown on screen. Please do not share it with anyone, including bank staff."]),
        ("CONTACT", "customer_service", "help",
         ["How can I contact customer service?",
          "What is the customer care number?",
          "What are your support hours?",
          "Is there a live chat with the bank?",
          "I need to speak to support about my account, how do I reach you?"],
         ["You can reach customer service by phone banking, through secure chat in the mobile app, or by email through the Contact Us section. Phone and chat support are available round the clock for urgent matters such as blocking a card, and other requests are handled during business hours. Please keep your customer ID handy so we can help you faster.",
          "There are several ways to contact us. The quickest is the chat option in the app, which is available all day. You can also call the number printed on the back of your card or on our website, or visit any branch during opening hours."]),
        ("CONTACT", "human_agent", "none",
         ["I want to talk to a real person.",
          "Can you transfer me to a human agent?",
          "I do not want a bot, please connect me to an advisor.",
          "I need a person to help me with this, not an automated reply.",
          "Please put me through to a customer service representative."],
         ["Of course. I will connect you with a human agent now. Please stay in this chat, and it helps if you briefly describe your issue so the agent can pick up straight away. Waiting times can vary, and if it is not urgent you can also request a call back at a time that suits you.",
          "Certainly, I can arrange for one of our advisors to help you. I am transferring you now, and I will share the details of our conversation so you do not need to repeat yourself. If the queue is long, you can choose to receive a call back instead."]),
        ("FIND", "find_ATM", "help",
         ["Where is the nearest ATM?",
          "Is there an ATM near {landmark}?",
          "I need to withdraw cash in {city}, where can I find one of your ATMs?",
          "Find me a cash machine close to me.",
          "Do you have ATMs that accept cash deposits nearby?"],
         ["You can find your nearest ATM using the ATM and Branch Locator in the app. Allow location access or type in an area such as {landmark}, and the map will show the closest machines, including ones that accept cash deposits. You can also use ATMs from partner networks with lower or no charges.",
          "To locate an ATM, open the Locator in the app or on our website and search by city or landmark. Filters help you find machines that offer deposits, cheque drops or extended hours. Please remember that using an ATM outside our network may attract a fee."]),
        ("FIND", "find_branch", "help",
         ["Where is the closest branch?",
          "What are the opening hours of the branch near {landmark}?",
          "Is there a branch in {city}?",
          "I need to visit a branch, can you help me find one?",
          "Which branch can I go to for a loan discussion?"],
         ["You can find branches using the Locator in the app or on our website. Search by city or a landmark like {landmark}, and you will see the address, opening hours and services available at each branch. Many branches allow you to book an appointment in advance, which reduces waiting time.",
          "To find a branch, enter your area or city in the Locator and choose Branches. Each listing shows the address, working hours and the services offered, such as loans or locker facilities. Most branches are open on weekdays and some are open on Saturday mornings."]),
    ],
    "payments_exception": [
        ("PAYMENT_EXCEPTION", "card_fraud_unauthorized_transaction", "escalate",
         ["There are transactions of {amount} on my card that I never made, I think my card was hacked.",
          "I see a charge from {merchant} for {amount} and I did not authorise it.",
          "Somebody used my {card_type} to buy things I do not recognise, this is fraud.",
          "I received an alert for {amount} at {merchant} while my card was in my wallet.",
          "My card ending {last4} has unknown purchases that appeared {when}, please help urgently."],
         ["I am very sorry this has happened, and I will treat it as urgent. First, block your card in the app so no further charges can be made. Then please tell us which transactions you do not recognise, including the {amount} at {merchant}, so we can raise a fraud claim. Our fraud team will investigate, may issue a replacement card, and will contact you with updates. Please do not share any codes or card details with anyone claiming to be from the bank.",
          "Thank you for reporting this quickly. Your card should be blocked immediately to prevent more misuse, and the disputed transactions will be flagged for our fraud team. They will review the details, and where a transaction is confirmed as fraudulent the amount is normally returned to your account after the investigation. A specialist will reach out to collect a few more details."]),
        ("PAYMENT_EXCEPTION", "duplicate_charge", "escalate",
         ["I was charged twice by {merchant} for the same purchase of {amount}.",
          "The same payment of {amount} shows up two times on my statement.",
          "Double debit on my card for {merchant}, please fix.",
          "I paid {amount} once but the money left my account twice {when}.",
          "Why did {merchant} take {amount} two times?"],
         ["I am sorry about the double charge. Sometimes one of the two entries is only a temporary hold that drops off within a few working days, so please check whether both are marked as completed. If both are completed, raise a dispute from the transaction details in the app for the duplicate {amount} at {merchant}. We will contact the merchant, and the extra amount will be returned if the duplicate is confirmed.",
          "Thanks for letting us know. Please keep proof of your purchase and, if possible, the merchant receipt. We will open a dispute for the duplicate payment, and a specialist will review it with the merchant. Refunds usually take several working days once the duplicate is confirmed."]),
        ("PAYMENT_EXCEPTION", "chargeback_request", "escalate",
         ["I never received the goods I paid {amount} for on {merchant}, I want a chargeback.",
          "The seller is not responding and I want my {amount} back through my card.",
          "I ordered from {merchant} {when}, the item is not what was shown, can I dispute the charge?",
          "How do I start a chargeback for a service that was not delivered?",
          "The merchant refused a refund of {amount}, what are my options with the bank?"],
         ["I am sorry about your experience. You can ask for a chargeback when goods or services were not delivered or were not as described. Please raise a dispute in the app for the {amount} payment to {merchant}, and attach evidence such as the order confirmation, messages with the seller and photographs if relevant. We will review your case, contact the merchant, and update you on the outcome.",
          "Thank you for the details. A card dispute can be raised for items not received or not as described. We will need your order confirmation and any communication with the seller. Our disputes team will assess the claim and let you know whether the amount can be recovered."]),
        ("PAYMENT_EXCEPTION", "transfer_pending_stuck", "escalate",
         ["My transfer of {amount} has been pending for {days} days.",
          "The money I sent {when} has not reached the receiver yet, the status still says processing.",
          "Payment stuck in processing for {days} days and my rent is due.",
          "I sent {amount} and it is not credited, but it was debited from my account.",
          "How long will my transfer stay pending? It has been {days} days."],
         ["I understand how worrying it is when a payment is delayed. Most transfers complete within one working day, but some can be held for security checks or because of details that need correcting. Please check the transaction status and reference number in the app. Since it has been {days} days, I will raise this as a payment investigation so a specialist can trace the payment with the receiving bank and update you.",
          "Sorry for the delay. A transfer can stay pending if the receiving bank is slow, if the details do not match, or if the payment is under review. I have noted the delay and a specialist team will trace the payment. If it cannot be completed, the money is returned to your account."]),
        ("PAYMENT_EXCEPTION", "transfer_wrong_account", "escalate",
         ["I entered the wrong account number and sent {amount}, please help me get it back.",
          "Sent {amount} to a stranger by mistake {when}.",
          "Made a transfer to the wrong beneficiary, how do I recover the money?",
          "The account number I typed was one digit off, and the payment went through.",
          "I think I paid the wrong person {amount} and I am panicking."],
         ["Please stay calm, and thank you for telling us quickly, because speed matters. Payments to a wrong account cannot always be reversed, but we can send a recall request to the receiving bank for the {amount}. Please share the transaction reference, the date and the account number you used. We will contact the other bank and, if the account holder agrees or the funds are still available, the money can be returned.",
          "I am sorry this happened. Please raise a wrong transfer request in the app or with customer service right away with the payment reference. We will ask the receiving bank to hold and return the funds, but they need the account holder consent, so the timing can vary. A specialist will follow up with you."]),
        ("PAYMENT_EXCEPTION", "international_transfer_compliance_hold", "escalate",
         ["My international transfer of {amount} to {city} is on hold and nobody tells me why.",
          "The bank asked for documents for a payment abroad and it is delayed.",
          "Why was my overseas payment blocked for a compliance review?",
          "My wire transfer has been under review for {days} days, I need the money urgently.",
          "Transfer to {city} flagged, what proof do I need to provide?"],
         ["I understand this is frustrating. International payments can be held for a routine compliance review, which is a regulatory requirement, and we may ask for supporting documents such as an invoice, contract or proof of source of funds. I cannot see the full details of the review in this chat, so I will pass your case to our payments compliance team, who will contact you about what is needed. Providing the documents promptly usually speeds things up.",
          "Thank you for your patience. Some cross border payments are checked before release, and I am not able to share the specific reason for a review. Please keep supporting documents ready. I am escalating your transfer to a specialist team, who will tell you what is required and the expected timing."]),
        ("PAYMENT_EXCEPTION", "card_declined_while_travelling", "escalate",
         ["My card was declined in {city} and I am stuck without money.",
          "I am abroad and my {card_type} stopped working at the shop, please help now.",
          "Card not working in {city}, I tried it {when} and it failed, I have no other way to pay.",
          "Payment declined overseas even though I have enough balance.",
          "Urgent, travelling and card blocked, I cannot pay my hotel."],
         ["I am sorry you are stuck, and I will help as quickly as possible. Cards are sometimes declined abroad because international usage is off, a limit has been reached, or a security check was triggered. Please open the app and check Card Controls for your {card_type}, and approve any security alert you received. If it still fails, I can connect you to our 24 hour support line so an agent can unblock the card or arrange emergency cash.",
          "Sorry about the inconvenience, especially while you are travelling. Please check that international transactions are enabled and that you have not reached your daily limit. Because this is urgent, I recommend calling the emergency line on the back of the card, where an agent can verify you and remove any block. Emergency cash or a replacement card can be arranged if needed."]),
        ("PAYMENT_EXCEPTION", "lost_card_abroad_emergency", "escalate",
         ["My wallet with my {card_type} was stolen in {city}, what do I do?",
          "I lost my card and passport in {city} and I have no cash.",
          "Card stolen abroad, need emergency replacement.",
          "Pickpocketed in {city} and all my cards are gone.",
          "I am in {city} and my {card_type} is lost, how can I get emergency help?"],
         ["I am sorry to hear that, and please make sure you are safe first. Block your cards immediately in the app, or call our 24 hour emergency line if you cannot log in. We can arrange an emergency replacement card or emergency cash in {city}, and our agent will verify your identity and guide you on the fastest option. I suggest reporting the theft to the local police as well, since some claims need a report.",
          "That sounds very stressful. The first step is to block all lost cards so nobody can use them. Then our emergency support team can help with a replacement card delivered to your location or a cash advance, after confirming your identity. Please keep any police report reference, as it can help with disputed transactions."]),
        ("PAYMENT_EXCEPTION", "recurring_payment_unauthorized", "escalate",
         ["{merchant} keeps charging me {amount} every month and I never subscribed.",
          "How do I stop a subscription that I cancelled but still gets billed {amount}?",
          "I found a recurring payment on my card that I do not recognise.",
          "Cancel the automatic payment to {merchant} right now.",
          "A free trial with {merchant} turned into a monthly charge of {amount}, help."],
         ["I understand how annoying unwanted charges are. You can stop a recurring payment by going to Cards and Recurring Payments in the app and choosing Stop for {merchant}. If the merchant has already taken money you did not agree to, we can raise a dispute for the {amount} charges. It also helps to contact the merchant to cancel the subscription so they stop trying to bill you.",
          "Sorry about the trouble. We can block future payments to the merchant on your card, and I can start a dispute for charges you did not authorise. Please share the dates and the amounts of the payments and any cancellation emails you have. A specialist will review whether the money can be recovered."]),
        ("PAYMENT_EXCEPTION", "large_transaction_blocked", "escalate",
         ["My payment of {big_amount} was blocked and I need it to go through today.",
          "The bank stopped my big payment for {big_amount}, why?",
          "I want to make a large purchase of {big_amount} but my card keeps getting declined.",
          "Transaction limit exceeded, can you increase my limit for today?",
          "Urgent: {big_amount} payment held for security, please release it."],
         ["I understand the urgency. Large payments can be blocked because they go over your daily limit or because our security systems flagged them for extra checks. I cannot release a held payment from this chat, but I can help you confirm it is genuine. Please approve any alert on your app, or I can connect you to a specialist who will verify you and, where possible, release the payment or raise your limit temporarily.",
          "Thank you for explaining. A large payment may exceed your set limit, or it may be waiting for extra verification. You can raise your limit in Card Controls after authentication, and the security team can release a held transaction after checking with you. Since it is time sensitive, shall I connect you to an agent now?"]),
        ("PAYMENT_EXCEPTION", "currency_conversion_dispute", "escalate",
         ["I was charged more than expected because of the exchange rate on my purchase in {city}.",
          "Why is the currency conversion fee on my {amount} payment so high?",
          "The amount debited does not match the price I saw abroad.",
          "Dispute foreign exchange markup on my card transaction.",
          "I paid in the local currency but was billed at a strange rate."],
         ["Thanks for raising this. When you pay in a foreign currency, the amount is converted at the network exchange rate on the day the transaction is settled, and a foreign currency fee may also apply, so the final figure can differ from what you saw at the shop. Please check the transaction details in the app for the rate and fee that were used. If you believe the amount is wrong, we can raise a query and a specialist will review it.",
          "I understand the surprise. The debited amount comes from the exchange rate at the time of settlement, plus any conversion charges, and the shop may also offer to charge in your home currency at their own rate. I can flag your transaction for review so a specialist can confirm whether the correct rate was applied."]),
        ("PAYMENT_EXCEPTION", "refund_not_received", "escalate",
         ["{merchant} said they refunded {amount} {days} days ago but I have not received it.",
          "My refund of {amount} is still missing, the shop says it was processed.",
          "Returned the item to {merchant} and still no money back.",
          "Where is my refund, it has been {days} days?",
          "I cancelled an order with {merchant} and the credit has not shown on my card."],
         ["I am sorry for the wait. Refunds from merchants usually take between five and ten working days to appear, depending on the merchant and the card network. Please ask {merchant} for the refund reference number or acquirer reference number, which lets us trace it. Once you share it, we can check whether the credit was received and, if not, raise an enquiry with the merchant bank.",
          "Thank you for the details. A refund can take some time after the merchant approves it. If it has been more than ten working days, please send us the refund confirmation from the merchant. We will trace the credit and follow up with the merchant bank on your behalf."]),
    ],
    "retail_exception": [
        ("RETAIL_EXCEPTION", "account_frozen_kyc", "escalate",
         ["My {account_type} has been frozen and I do not know why.",
          "I cannot make any transactions, the bank says KYC is pending.",
          "Account blocked after I updated my address, how do I get access?",
          "Why is my account restricted, I need to pay bills today.",
          "Please unfreeze my account, I already submitted my documents {when}."],
         ["I understand how stressful it is to have an account restricted. Accounts are sometimes frozen when identity documents are missing or expired, or when a compliance review is needed. I cannot lift a restriction from this chat, but I can raise a priority request with our account services team. Please keep your photo ID, proof of address and recent photograph ready, and a specialist will contact you with the exact documents required.",
          "I am sorry for the inconvenience. A restriction usually means we need to verify or update some information. You can complete the update in the app under Profile and KYC, or visit a branch. If you have already submitted documents, I will ask the team to prioritise the review and update you."]),
        ("RETAIL_EXCEPTION", "deceased_account_holder", "escalate",
         ["My father passed away recently, how do I access his {account_type}?",
          "I need to inform the bank about the death of my husband and settle his accounts.",
          "What is the process to claim the balance in my late mother's account?",
          "How do I close the account of a deceased family member?",
          "My brother died and I am the nominee on his account, what should I do?"],
         ["I am very sorry for your loss. Please take your time, and we will make this as simple as we can. To begin, please inform us of the death by contacting our bereavement team, who will guide you personally. You will usually need a copy of the death certificate, your own identification and proof of your relationship, and if you are the nominee, the nomination details. Because this involves legal steps, a specialist will handle your request and explain the timeline.",
          "My condolences to you and your family. Accounts of a deceased customer are handled by a dedicated team with care. Please share the death certificate and your identification, and they will explain how the balance is released to the nominee or legal heir and how the account is closed. I will connect you with them, so you do not have to repeat yourself."]),
        ("RETAIL_EXCEPTION", "joint_account_dispute", "escalate",
         ["My spouse emptied our joint account without telling me.",
          "We are separating and I want to stop my partner from withdrawing from our joint account.",
          "How can I remove someone from a joint account?",
          "The other holder of my joint {account_type} is not cooperating, what are my options?",
          "I want to freeze the joint account until we settle things."],
         ["I am sorry you are going through this. Joint accounts are governed by the mandate that was agreed when they were opened, and changes such as removing a holder or freezing the account usually need the consent of all holders, or a legal instruction. I can arrange for a specialist to explain your options and what proof would be required. Please keep in mind that we cannot take sides, and for serious disputes it can help to seek legal advice.",
          "Thank you for explaining. Changing the operating instruction on a joint account depends on the mandate, and sometimes a court order is needed. Our specialist team can review the account terms with you and tell you what can be done, such as requiring both signatures for withdrawals. I will connect you with them."]),
        ("RETAIL_EXCEPTION", "loan_hardship_request", "escalate",
         ["I lost my job and cannot pay my {loan_type} instalments.",
          "I need help because I am struggling to repay my loan of {loan_amount}.",
          "Can I get a payment holiday on my loan for a few months?",
          "I missed {days} instalments and I am afraid of penalties.",
          "My income has dropped, can you restructure my {loan_type}?"],
         ["I am sorry you are facing this, and I am glad you reached out early. We have options for customers in financial difficulty, such as a temporary payment holiday, a longer tenure with lower instalments, or restructuring, depending on your situation. I will pass your request to our customer support team, who will review your income and expenses and discuss what is possible. Please try to keep making any payments you can, and avoid ignoring reminders, as talking to us early gives you more choices.",
          "Thank you for being open about this. We treat hardship requests seriously and with respect. A specialist will contact you to understand your circumstances and explore ways to reduce the pressure, for example a payment break or revised repayment plan. They may ask for proof of the change in your income. Nothing is decided in this chat, but I will raise it as a priority."]),
        ("RETAIL_EXCEPTION", "mortgage_arrears_support", "escalate",
         ["I am behind on my mortgage payments and worried about losing my home.",
          "We fell {weeks} weeks behind on the mortgage, what can we do?",
          "The bank sent a notice about mortgage arrears, please help.",
          "How can I avoid repossession of my house?",
          "I cannot afford my mortgage after the rate increase."],
         ["I understand how worrying this is, and I want you to know that help is available. Please do not ignore the notice, because early contact gives us more options such as revising the payment plan, extending the term or a temporary arrangement. I will connect you with our specialist mortgage support team, who can go through the details with you in confidence. It is also worth getting free independent money advice alongside speaking to us.",
          "Thank you for reaching out. Falling behind on a mortgage is stressful, and there are steps we can explore together. A mortgage support specialist will review your account, your income and expenses, and discuss a plan that suits you. I am raising this as urgent so they contact you soon."]),
        ("RETAIL_EXCEPTION", "identity_theft_account_takeover", "escalate",
         ["Someone changed my phone number and email on my account and I am locked out.",
          "I think my identity was stolen, there are accounts and loans in my name that I did not open.",
          "My online banking was accessed by someone else and money is missing.",
          "I got a message that my password was changed and I did not do it.",
          "Suspicious login on my account from another country, please secure it."],
         ["I am very sorry, and I will treat this as urgent. To protect you, we should restrict access to your account and cards immediately, and I am escalating this to our fraud and security team now. Please do not use the same password anywhere else, avoid clicking links in unexpected messages, and keep a note of any suspicious messages or calls. A specialist will contact you on a verified number to restore your access and review the activity.",
          "Thank you for flagging this quickly. Access to your online banking will be locked while we investigate. Our security team will verify your identity through a secure process, review recent changes and transactions, and help reset your details. Please tell us about any transaction you did not make, so it can be investigated."]),
        ("RETAIL_EXCEPTION", "power_of_attorney_request", "escalate",
         ["My mother is unwell and I need to manage her account, how does power of attorney work?",
          "I have a power of attorney document, how do I register it with the bank?",
          "Can my son operate my account while I am in hospital?",
          "How do I authorise someone to handle my banking on my behalf?",
          "The account holder cannot sign anymore, what documents do I need to act for them?"],
         ["I am sorry to hear that. Managing another person's account requires a valid legal authority, such as a registered power of attorney, and we need to verify it before giving access. Please prepare the original or certified copy of the document, your identification and the account holder details. A specialist will check the paperwork and explain what your authority allows, and I will arrange for them to contact you."],
        ),
        ("RETAIL_EXCEPTION", "legal_name_change", "escalate",
         ["I changed my name after marriage, how do I update it on my {account_type}?",
          "My legal name has changed and my cards still show the old one.",
          "What documents do I need to update my name with the bank?",
          "Please change my surname on all my accounts.",
          "Name mismatch is blocking my loan, how do I correct it in your records?"],
         ["To change your name in our records, we need proof of the legal name change, such as a marriage certificate, a gazette notification or a deed poll, together with your identification. Please upload the documents in the app under Profile or visit a branch, and we will update your accounts and cards. Because this affects several products, the update may take a few working days, and we will notify you when it is complete."]),
        ("RETAIL_EXCEPTION", "overdraft_fee_dispute", "escalate",
         ["I was charged an overdraft fee of {amount} even though I had funds.",
          "Why is my {account_type} overdrawn? I never spent that much.",
          "Please waive the overdraft charges, they are unfair.",
          "My salary came late and I was penalised, can you reverse the fee?",
          "I disagree with the unarranged overdraft fee on my statement."],
         ["I understand your concern, and I will help you look into it. Overdraft fees can arise when a payment goes through before a deposit is credited, or when the balance goes below the agreed limit. Please check the dates on your statement for the fee of {amount}. If you feel it was applied unfairly, we can review it as a goodwill request or a complaint, and a specialist will assess your account history.",
          "Thanks for explaining. To review the fee, we need to see the order in which the deposit and the payments were processed. I will raise a request for the fee to be reviewed, and you will hear back with the outcome. In some cases, especially after a delayed salary, the fee can be waived."]),
        ("RETAIL_EXCEPTION", "formal_complaint_escalation", "escalate",
         ["I have complained three times and nobody has solved my problem, I want to escalate.",
          "How do I file a formal complaint against the bank?",
          "I am not satisfied with the resolution and want to take this to the ombudsman.",
          "Please escalate my complaint to a senior manager.",
          "I want a written response to my complaint about {account_type} charges."],
         ["I am sorry that your issue has not been resolved, and you have every right to escalate. I can register a formal complaint now, which will be given a reference number and reviewed by our complaints team, who will respond in writing within the time required. If you are not satisfied with the final response, you can take the matter to the external financial ombudsman. It helps if you share the dates and details of your previous contacts."]),
        ("RETAIL_EXCEPTION", "loan_application_rejected_appeal", "escalate",
         ["My loan application for {loan_amount} was rejected and I do not understand why.",
          "How can I appeal a declined {loan_type}?",
          "The bank refused my mortgage even though I have a stable income.",
          "Can I reapply after a rejection and what should I improve?",
          "Please tell me the reason my loan was denied."],
         ["I am sorry that your application was not approved. Decisions are based on factors such as income, existing debts, credit history and the value of the loan requested, and we can share the main reasons with you. You may ask for a review, especially if your circumstances have changed or if some information was missing. I will pass your request to a loans specialist who can explain the decision and suggest what may improve your chances next time."]),
        ("RETAIL_EXCEPTION", "vulnerable_customer_support", "escalate",
         ["I am elderly and find the app very hard to use, can someone help me with my banking?",
          "I have a visual impairment and cannot read the statements, what support do you offer?",
          "I am confused and worried about a scam call asking for my details.",
          "I need extra help managing my money after my illness.",
          "Can I get someone to visit me at home for banking services?"],
         ["Thank you for telling us, and please know that we are here to help in a way that works for you. We offer support such as large print or audio statements, assisted banking at branches, a dedicated helpline with priority handling and, in some areas, home visits. If you are worried about a suspicious call, please do not share any codes or personal details, and we will help you check it. I will connect you with our customer care specialists to set up the support you need."]),
    ],
}

# Convention check: the last response body of every intent must be slot-free.
for _group, _items in INTENTS.items():
    for _cat, _intent, _tone, _instrs, _resps in _items:
        assert not fields(_resps[-1]), f"{_intent}: last response body must have no slots"


def build_row(rng, group):
    category, intent, tone, instrs, resps = rng.choice(INTENTS[group])
    slots = make_slots(rng)
    template = rng.choice(instrs)
    usable = [r for r in resps if fields(r) <= fields(template)]
    body = rng.choice(usable) if usable else resps[-1]
    instruction = style(rng, template.format(**slots))
    response = build_response(rng, tone, body.format(**slots))
    return instruction, category, intent, response


def random_mix(rng):
    w = [rng.uniform(0.28, 0.42), rng.uniform(0.28, 0.42), rng.uniform(0.08, 0.22), rng.uniform(0.08, 0.22)]
    total = sum(w)
    return [x / total for x in w]


def main():
    rng = random.Random(SEED)
    seen = set()
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []

    for i in range(1, N_FILES + 1):
        mix = random_mix(rng)
        counts = [int(round(ROWS_PER_FILE * p)) for p in mix]
        counts[-1] += ROWS_PER_FILE - sum(counts)

        rows = []
        for group, count in zip(GROUPS, counts):
            for _ in range(count):
                for _attempt in range(500):
                    row = build_row(rng, group)
                    if row[0] not in seen:
                        seen.add(row[0])
                        rows.append(row)
                        break
                else:
                    raise RuntimeError(f"Could not find a unique instruction for {group}")
        rng.shuffle(rows)

        path = f"{OUT_DIR}/synthetic_bitext_{i:02d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        manifest.append([os.path.basename(path), len(rows)] + counts)
        print(f"Wrote {path}: {len(rows)} rows  (payments={counts[0]}, retail={counts[1]}, "
              f"payment_exc={counts[2]}, retail_exc={counts[3]})")

    with open(f"{OUT_DIR}/synthetic_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "rows"] + GROUPS)
        writer.writerows(manifest)
    print(f"\nDone: {N_FILES} files, {sum(m[1] for m in manifest)} rows total, all instructions unique.")


if __name__ == "__main__":
    main()
