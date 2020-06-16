import numpy as np
import pandas as pd


class dataCleaner_new:
    def __init__(self, columns, length_return, topK, start_date, end_date, split_rate):
        self.columns = columns
        self.num_days = 0
        self.num_observation = 0
        self.length_return = length_return
        self.topK = topK
        self.start_date = start_date
        self.end_date = end_date
        self.split_rate = split_rate

    # About Stock return
    def make_return_price(self):
        data = pd.read_csv("Price.csv")
        data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y")

        prices = data["Close"].values
        returns = ((data["Close"].values[1:] - data["Close"].values[:-1]) / data["Close"].values[:-1]).tolist()
        returns.insert(0, "No")

        returns_vec = []

        for i in range(self.length_return, len(returns)):
            returns_vec.append(returns[i - self.length_return + 1: i + 1])
        dates = data["Date"].values[self.length_return:]

        return_dict = {}
        for j in range(len(dates)):
            return_dict[dates[j]] = returns_vec[j]

        price_dict = {}
        full_dates = data["Date"].values
        for k in range(len(dates)):
            price_dict[dates[k]] = prices[k]

        return return_dict, price_dict

    # About risk free interest rate
    def make_risk(self):
        risk_data = pd.read_csv("Risk.csv")
        risk_data["DATE"] = pd.to_datetime(risk_data["DATE"], format="%Y-%m-%d")
        dates = risk_data["DATE"].values
        risks = risk_data["RISK"].values

        for i in range(len(risks)):
            if risks[i] == ".":
                risks[i] = risks[i - 1]

        risks = risks.astype(np.float) / 100

        return_dict = {}
        for i in range(len(dates)):
            return_dict[dates[i]] = risks[i]
        return return_dict

    def make_final_data(self, price_data, return_data, risk_data):
        option_data = pd.read_csv("BigOptionData.csv")[self.columns]

        option_data["date"] = pd.to_datetime(option_data["date"], format="%Y/%m/%d")
        option_data["exdate"] = pd.to_datetime(option_data["exdate"], format="%Y/%m/%d")

        option_data["T"] = option_data["exdate"] - option_data["date"]
        option_data["strike_price"] = option_data["strike_price"] / 1000
        option_data["spotclose"] = (option_data["best_bid"] + option_data["best_offer"]) / 2
        option_data.drop(columns=["best_bid", "best_offer"])
        option_data = option_data[option_data["date"] >= self.start_date]
        option_data = option_data[option_data["date"] <= self.end_date]

        all_days = option_data["date"].unique()
        split_date = all_days[int(len(all_days) * self.split_rate)]

        train_ = option_data[option_data["date"] < split_date]
        test_ = option_data[option_data["date"] >= split_date]

        # a list of trading days datetime objects in the every month
        train_keys = train_["date"].unique()
        test_keys = test_["date"].unique()

        train = {}
        for i, p in enumerate(train_keys):
            day_option_data = train_[train_["date"] == p]
            day_option_data = day_option_data.nlargest(self.topK, "volume")
            day_price_data = price_data[p]
            day_return_data = return_data[p]
            day_risk_data = risk_data[p]

            train[p] = [day_option_data, day_risk_data, day_return_data, day_price_data]

        test = {}
        for i, p in enumerate(test_keys):
            day_option_data = test_[test_["date"] == p]
            day_option_data = day_option_data.nlargest(self.topK, "volume")
            day_price_data = price_data[p]
            day_return_data = return_data[p]
            day_risk_data = risk_data[p]

            test[p] = [day_option_data, day_risk_data, day_return_data, day_price_data]

        # add processed data info
        self.num_train_days = len(train_keys)
        self.num_train_obs = self.num_train_days * self.topK
        self.num_test_days = len(test_keys)
        self.num_test_obs = self.num_test_days * self.topK

        print("Attached {} days of train data".format(self.num_train_days))
        print("Attached {} train option observations".format(self.num_train_obs))
        print("Attached {} days of test data".format(self.num_test_days))
        print("Attached {} test option observations".format(self.num_test_obs))

        return train, test

    def get_train_num(self):
        return self.num_train_days, self.num_train_obs

    def get_test_num(self):
        return self.num_test_days, self.num_test_obs
