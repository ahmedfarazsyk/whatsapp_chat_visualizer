import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    data = bytes_data.decode("utf-8")

    df = preprocessor.preprocess(data)

    #fetch unique users
    user_list = df["user"].unique().tolist()
    user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis w.r.t ", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, num_words, num_media_files, num_of_links = helper.fetch_stats(selected_user, df)
        
        st.title("Top Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(num_words)

        with col3:
            st.header("Total Media Files")
            st.title(num_media_files)

        with col4:
            st.header("Total Links")
            st.title(num_of_links)

        #Sentiment Analysis
        st.title("Sentiment Analysis")
        positive, neutral, negative = helper.sentiment_analysis(selected_user, df)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Positive Messages")
            st.title(positive)

        with col2:
            st.header("Neutral Messages")
            st.title(neutral)

        with col3:
            st.header("Negative Messages")
            st.title(negative)


        # Monthly Timeline
        st.title("Monthly Timeline")
        month_timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()

        ax.plot(month_timeline["time"], month_timeline["message"], color = "g")
        plt.xticks(rotation = 90)
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        day_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()

        ax.plot(day_timeline["only_date"], day_timeline["message"], color = "g")
        plt.xticks(rotation = 90)
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation = 90)
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color = "orange")
            plt.xticks(rotation = 90)
            st.pyplot(fig)

        # Activity Heatmap
        st.title("Weekly Activity Map")
        pivot_map = helper.activity_heatmap(selected_user, df)

        fig, ax = plt.subplots()
        ax = sns.heatmap(pivot_map)
        st.pyplot(fig)


        #finding the most bussiest users in a group(Group Level)
        if selected_user == "Overall":
            st.title("Most Busy Users")

            col1, col2 = st.columns(2)

            x, new_df = helper.fetch_most_busy_users(df)
            fig, ax = plt.subplots()

            with col1:
                ax.bar(x.index, x.values, color = "r")
                plt.xticks(rotation = 45)
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)


        # Generating wordcloud
        st.title("Word cloud")
        
        try:
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
        except:
            df_wc = helper.create_wordcloud(selected_user, pd.DataFrame({"user":[selected_user],"message":["(No-messages)"]}))
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

        # most common words
        st.title("Most common Words")
        try:
            common_words = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots()
            ax.barh(common_words[0], common_words[1])
            st.pyplot(fig)
        except:
            common_words = helper.most_common_words(selected_user, pd.DataFrame({"user":[selected_user],"message":["(No-messages)"]}))
            fig, ax = plt.subplots()
            ax.barh(common_words[0], common_words[1])
            st.pyplot(fig)


        #emoji analysis
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        emoji_df  = helper.emoji_analysis(selected_user, df)

        with col1: 
            st.dataframe(emoji_df)

        with col2:
            fig, ax = plt.subplots()

            try:
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct = "%0.2f")
            except:
                ""

            st.pyplot(fig)

        