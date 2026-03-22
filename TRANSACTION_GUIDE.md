# Manual Transaction Input Guide

## How to fill in the fields

### Transaction Type
The type of transaction encoded as a number.

| Value | Type | Can be fraud? |
|---|---|---|
| 0 | CASH_IN | No |
| 1 | CASH_OUT | Yes |
| 2 | DEBIT | No |
| 3 | PAYMENT | No |
| 4 | TRANSFER | Yes |

> Fraud only ever appears on **TRANSFER** and **CASH_OUT** transactions in this dataset.

---

### Step
The hour at which the transaction occurred in the simulation (1 step = 1 hour).
- Range: 1 – 744 (31 days)
- For manual testing, just leave it as `1`

---

### Amount
The value of the transaction in dollars.
- Fraud transactions tend to be **large amounts** that equal the sender's full balance

---

### Sender — Before & After

| Field | Meaning |
|---|---|
| Sender — before | Sender's balance before the transaction |
| Sender — after | Sender's balance after the transaction |

**Rule:** `Sender after = Sender before − Amount`

**Fraud signal:** Sender after = **exactly 0** (account fully drained)

---

### Receiver — Before & After

| Field | Meaning |
|---|---|
| Receiver — before | Receiver's balance before the transaction |
| Receiver — after | Receiver's balance after the transaction |

**Rule (normal):** `Receiver after = Receiver before + Amount`

**Fraud signal:** Receiver after = **same as Receiver before** (money never arrives — it was immediately moved again, a classic money laundering pattern called *layering*)

---

### System Fraud Flag
Set by the bank's internal system for very large transfers. In practice this feature has weak signal — leave it as **No (0)** for most tests.

---

## Scenario Templates

### ✅ Legitimate Payment
```
Type:             PAYMENT
Amount:           4,878
Sender before:    170,136
Sender after:     165,258      ← before − amount
Receiver before:  0
Receiver after:   4,878        ← before + amount
Flag:             No
```

### ✅ Legitimate Transfer
```
Type:             TRANSFER
Amount:           10,000
Sender before:    50,000
Sender after:     40,000       ← before − amount
Receiver before:  5,000
Receiver after:   15,000       ← before + amount
Flag:             No
```

### 🚨 Fraudulent Transfer (account drain + layering)
```
Type:             TRANSFER
Amount:           450,000
Sender before:    450,000
Sender after:     0            ← fully drained ⚠
Receiver before:  0
Receiver after:   0            ← money disappeared ⚠
Flag:             No
```

### 🚨 Fraudulent Cash Out
```
Type:             CASH_OUT
Amount:           200,000
Sender before:    200,000
Sender after:     0            ← fully drained ⚠
Receiver before:  0
Receiver after:   0            ← money disappeared ⚠
Flag:             No
```

---

## The Two Fraud Red Flags

```
1. Sender after = 0          (account completely emptied)
2. Receiver after = 0        (funds never actually landed — moved on immediately)
```

When both of these are true on a TRANSFER or CASH_OUT,
the model will flag it as fraud with very high confidence (>99%).
