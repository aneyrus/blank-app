import pandas as pd
import numpy as np
import streamlit as st
from util_functions import calculate_statistics,  plot_graph, fetch_and_save_player_props
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="Player Props Metrics")

st.markdown(
    """
    <style>
    @media print {
        /* Remove sidebar during print */
        .stSidebar {
            display: none;
        }

        /* Remove header and footer during print */
        header, footer {
            display: none;
        }

        /* Remove margins and padding from the body and app container */
        .stApp, .main {
            margin: 0;
            padding: 0;
            width: 100%;
        }

        /* Force everything to fit on one page */
        @page {
            size: A4;
            margin: 0;
        }

        /* Reduce all unnecessary spacing and padding between elements */
        .block-container {
            padding: 0;
            margin: 0;
        }

        .stText, .stMarkdown, .stImage, .stTable, .stDataFrame {
            margin: 0;
            padding: 0;
        }

        .stTitle, .stHeader, .stSubheader, .stText, .stMarkdown {
            font-size: 12px !important;  /* You can adjust this to make it smaller */
        }

        img {
            max-width: 100%;
            height: auto;
        }

        /* Hide buttons and widgets to save space */
        .stButton {
            display: none;
        }

        /* Remove any other unwanted elements */
        .stDownloadButton {
            display: none;
        }

        /* Make sure no extra page breaks */
        .block-container {
            page-break-inside: avoid;
        }
    }
    </style>
    """, unsafe_allow_html=True
)

by_game = st.sidebar.toggle("Odds Compare")
sport = st.sidebar.selectbox("Select a Sport", ['NBA','NFL'])


if sport == 'NFL':
    @st.cache_data 
    def load_nfl():
        file_path = "01-26-NFL.xlsx"
        df = pd.read_excel(file_path)
        df.columns = [col.replace("\n", "_").replace(" ", "_").upper() for col in df.columns]
        df.rename(columns={'PLAYER': 'Player Name', 'WEEK_#': 'WEEK', 'PASSING_COMP': 'Passing Completions',
            'PASSING_ATT': 'Passing Attempts', 'PASSING_YDS': 'Passing Yards','PASSING_TD': 'Passing Touchdowns',
            'PASSING_INT': 'Passing Interceptions', 'PASSING_SK': 'Passing Sacks', 'PASSING_LG': 'Longest Pass',
            'RUSHING_ATT': 'Rushing Attempts', 'RUSHING_YDS': 'Rushing Yards', 'RUSHING_TD': 'Rushing Touchdowns',
            'RUSHING_LG': 'Longest Rush', 'RECEIVING_TAR': 'Receiving Targets', 'RECEIVING_REC': 'Receiving Receptions',
            '_RECEIVINGYDS': 'Receiving Yards','RECEIVING_TD': 'Receiving Touchdowns', 'RECEIVING_LG': 'Longest Reception'
        }, inplace=True)
        return df
    df = load_nfl()
    stat_options = [
        'Passing Completions',
        'Passing Attempts',
        'Passing Yards',
        'Passing Touchdowns',
        'Passing Interceptions',
        'Passing Sacks',
        'Longest Passes',
        'Rushing Attempts',
        'Rushing Yards',
        'Rushing Touchdowns',
        'Longest Rush',
        'Receiving Targets',
        'Receiving Receptions',
        'Receiving Yards',
        'Receiving Touchdowns',
        'Longest Reception'
    ]

if sport == 'NBA':
    def load_nba():
        uploaded_file = st.sidebar.file_uploader("Upload an NBA Excel File", type=["xlsx"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, engine="openpyxl")  # Read uploaded file
        else:
            st.warning("Please upload an Excel file.")
        df.columns = [col.replace("\n", "_").replace(" ", "_").upper() for col in df.columns]
        df.rename(columns={'FG': 'Field Goals', 'FGA': 'Field Goal Attempts', '3P': 'Three-Point Field Goals',
                 '3PA': 'Three-Point Field Goal Attempts', 'FT': 'Free Throws','FTA': 'Free Throw Attempts',
                 'OR': 'Offensive Rebounds', 'DR': 'Defensive Rebounds', 'TOT': 'Total Rebounds',
                 'A': 'Assists', 'PF': 'Personal Fouls', 'ST': 'Steals', 'TO': 'Turnovers',
                 'BL': 'Blocks', 'PTS': 'Points','PLAYER__FULL_NAME': 'Player Name',
        }, inplace=True)
        return df
    df = load_nba()
    df['WEEK'] = df.groupby('Player Name').cumcount() + 1
    nba_schedule = pd.read_csv("nba_schedule.csv")
    nba_schedule["Game Date"] = pd.to_datetime(nba_schedule["Game Date"], format="%m/%d/%Y")
    today = date.today() - timedelta(days=0)
    today_games = nba_schedule[nba_schedule["Game Date"].dt.date == today]
    @st.cache_data
    def fetch_and_process_odds(today_games):
        odds = fetch_and_save_player_props(
        today_games["Home/Neutral"].tolist(),
        "basketball_nba",
        "player_points,player_assists,player_rebounds,player_steals,player_blocks,player_points_alternate,player_assists_alternate,player_rebounds_alternate",
        )
        return odds
    odds = fetch_and_process_odds(today_games)
    games_list = [f"{visitor} at {home}" for visitor, home in zip(today_games["Visitor/Neutral"], today_games["Home/Neutral"])]
    stat_options = ['Total Rebounds', 'Assists', 'Steals', 'Blocks', 'Points']

stat = st.sidebar.selectbox("Select a Metric", stat_options)


if by_game:
    odds['market_key'] = odds['market_key'].replace({
    'player_points': 'Points',
    'player_assists': 'Assists',
    'player_rebounds': 'Total Rebounds',
    'player_steals' : 'Steals',
    'player_blocks' : 'Blocks',
    'player_points_alternate' : 'Points',
    'player_assists_alternate' : 'Assists',
    'player_rebounds_alternate' : 'Total Rebounds',})   
    odds = odds[(odds['market_key'].str.contains(stat, case=False, na=False)) & (odds['outcome_name'] == "Over")]
    allowed_bookmakers = ["FanDuel", "BetMGM", "DraftKings"]
    odds = odds[odds["bookmaker_title"].isin(allowed_bookmakers)]
    odds = odds.sort_values(by=['market_key', 'outcome_name', 'outcome_point'], ascending=[True, True, True])
    home_team = today_games["Home/Neutral"].dropna().astype(str).unique()
    filtered_odds = odds[odds["home_team"].isin(home_team)]
    players = filtered_odds["outcome_description"].unique()


else:
    players = [st.sidebar.selectbox("Select a Player", df["Player Name"].unique())]
    confidence_interval = st.sidebar.selectbox("Select Confidence Interval", options=[90, 95, 99], index=1 )

max_week = df['WEEK'].max()
num_games = st.sidebar.selectbox("Only Look at # of Last Games",  options=["Full Season"] + [str(i) for i in range(1, max_week + 1)])
filtered_data = pd.DataFrame()

for player in players:
    filtered_df = df[df["Player Name"] == player][["Player Name", "WEEK", "DATE", stat]].dropna()
    
    if num_games != "Full Season":
        num_games = int(num_games)
        player_week = filtered_df['WEEK'].max()
        last_weeks = filtered_df[filtered_df['WEEK'] >= (player_week - num_games + 1)]['WEEK']
        filtered_df = filtered_df[filtered_df['WEEK'].isin(last_weeks)]
    
    # Append to filtered_data DataFrame instead of dictionary
    filtered_data = pd.concat([filtered_data, filtered_df], ignore_index=True)



all_results = []  # Store results for all bet filters

if by_game:
    bet_filters_list = odds['outcome_point'].drop_duplicates().tolist()
    all_results = []  # Store all dataframes before concatenation

    for bet_filter in bet_filters_list:  # Loop through all bet filters
        hit_percentage_list = []
        
        for player in filtered_data["Player Name"].unique():
            player_data = filtered_data[filtered_data["Player Name"] == player]
            total_games = len(player_data)
            games_hit = (player_data[stat] >= bet_filter).sum()
            hit_percentage_value = (games_hit / total_games * 100) if total_games > 0 else 0
            hit_percentage_list.append({
                "Player Name": player,
                "Stat": stat,
                "Hit Percentage": hit_percentage_value,
                "Bet Filter": bet_filter  # Add bet filter for reference
            })

        hit_percentage_df = pd.DataFrame(hit_percentage_list)
        hit_percentage_df = hit_percentage_df.sort_values(by="Hit Percentage", ascending=False)

        all_results.append(hit_percentage_df)  # Append hit percentage data

    # Combine all hit percentage data before merging with pivoted odds
    final_hit_percentage_df = pd.concat(all_results, ignore_index=True)
    
    filtered_odds = filtered_odds.rename(columns={'outcome_description': 'Player Name'})
    pivoted_odds = filtered_odds.pivot_table(
        index=['Player Name', 'market_key', 'outcome_point'], 
        columns='bookmaker_title', 
        values='outcome_price', 
        aggfunc='first'
    ).reset_index()
    
    pivoted_odds.columns.name = None

    final_df = pd.merge(final_hit_percentage_df, pivoted_odds,
                        left_on=["Player Name", "Bet Filter", "Stat"],  
                        right_on=["Player Name", "outcome_point", "market_key"],
                        how="left"  )
    final_df['Hit Percentage'] = final_df['Hit Percentage'].apply(lambda x: f"{x:.2f}%")
    final_df["Hit Percentage"] = pd.to_numeric(final_df["Hit Percentage"].str.replace('%', '', regex=True), errors='coerce')
    final_df = final_df.sort_values(by="Hit Percentage", ascending=False)
    final_df = final_df.dropna(subset=["FanDuel"])
    final_df = final_df.drop(columns=["market_key", "outcome_point"])
    def calculate_ev(row):
        hit_percentage = row["Hit Percentage"] / 100  
        odds = row["FanDuel"]  #
        
        if np.isnan(odds): 
            return 0
        if odds < 0:
            odds_factor = (100 / abs(odds)) + 1
        elif odds > 0:
            odds_factor = (odds / 100) + 1 
        else:
            odds_factor = 0  
        ev = (odds_factor * hit_percentage) - (1 - hit_percentage)
        return round(ev, 3)
    final_df["EV"] = final_df.apply(calculate_ev, axis=1)
    sort_cust = st.sidebar.selectbox("Sort By", ['Hit Percentage','EV'])
    final_df = final_df.sort_values(by=sort_cust, ascending=False)
    st.header(f"Player % with {stat} Over Last {num_games} Games (Multiple Bet Filters)")
    html_table = final_df.to_html(classes='styled-table', index=False)

    css = """
        <style>
            .streamlit-expanderHeader {
                display: block;
            }
            .styled-table {
                width: 80%;  /* Set width to a reasonable value */
                margin: 25px 10px;
                border-collapse: collapse;
                font-family: 'Arial', sans-serif;
                text-align: left;
                margin-left: 10px; /* Aligns the table to the left */
                margin-right: 10px; /* Remove margin on the right */
                float: left; /* Force the table to float left */
            }
            .styled-table th,
            .styled-table td {
                padding: 12px;
                border: 1px solid #ddd;
            }
            .styled-table th {
                background-color: #003366;  /* Darker blue */
                color: white;
                font-weight: bold;
                text-align: center;
            }
            .styled-table td {
                text-align: center;
            }
            .styled-table tr:hover {
                background-color: #f5f5f5;
            }
        </style>
    """

    # Add custom CSS styles to Streamlit
    st.markdown(css, unsafe_allow_html=True)

    # Display the styled HTML table in Streamlit
    st.markdown(html_table, unsafe_allow_html=True)

else:
    num_options = np.arange(0, 2.1, 0.1)  
    default_value = 0.5
    default_index = list(num_options).index(default_value)
    graph_spacing = st.sidebar.selectbox("Use to give spacing from points to text", num_options, index = default_index)

    try:
        stat_metric = stat.split()[1]
    except IndexError:
        stat_metric = stat

    
    if num_games != "Full Season":
        if sport == "NFL": 
            st.header(f"{player} {stat} Last {num_games} Weeks ({last_weeks.min()} - {last_weeks.max()})")
            time_blurb = f"the last {num_games} weeks"
        else:
            st.header(f"{player} {stat} Last {num_games} Games")
            time_blurb = f"the last {num_games} games"       
    else:
        st.markdown(
            """
            <style>
            /* Adjust the "Full Season" header size, make it bold, and remove margin */
            .small-header {
                font-size: 14px !important;  /* Adjust this to make it smaller */
                font-weight: bold;           /* Make the text bold */
                display: inline;             /* Keep the "Full Season" on the same line */
                margin-top: 0px;             /* Remove top margin */
                margin-bottom: 0px;         /* Remove bottom margin */
                padding-top: 0px;            /* Remove any internal padding */
                padding-bottom: 0px;        /* Remove internal padding */
            }
            
            /* Adjust the size of the st.header element itself */
            .stHeader {
                font-size: 14x !important;  /* Make the header text smaller */
                margin-bottom: 0px !important; /* Remove bottom margin */
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.header(f"{player} {stat}")
        time_blurb = "the Full Season"


    mean, min_val, max_val, std_dev, ci_lower, ci_upper = calculate_statistics(filtered_data, stat,confidence_interval)

    benchamrk = st.sidebar.text_input('Enter Spread To Beat:')

    blurb = f"""
    Upon analyzing {player}'s {stat} across {time_blurb}, we observe a clear clustering around {mean:.0f} {stat_metric.lower()}. 
    When examining the entire season, the data reveals a standard deviation of {std_dev:.2f}, with a {confidence_interval}% confidence interval 
    ranging from {ci_lower:.2f} - {ci_upper:.2f}. This provides compelling evidence supporting a strong likelihood of exceeding {benchamrk} {stat_metric.lower()}.
    """

    st.markdown(blurb)


    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Use markdown for larger labels and smaller values with HTML
    col1.markdown('<p style="font-size: 16px; text-align: center; margin-bottom: 2px;">Average</p>', unsafe_allow_html=True)
    col1.markdown(f"<p style='font-size: 18px; text-align: center; margin-top: 0;'>{mean:.2f}</p>", unsafe_allow_html=True)

    col2.markdown('<p style="font-size: 16px; text-align: center; margin-bottom: 2px;">Min</p>', unsafe_allow_html=True)
    col2.markdown(f"<p style='font-size: 18px; text-align: center; margin-top: 0;'>{min_val:.2f}</p>", unsafe_allow_html=True)

    col3.markdown('<p style="font-size: 16px; text-align: center; margin-bottom: 2px;">Max</p>', unsafe_allow_html=True)
    col3.markdown(f"<p style='font-size: 18px; text-align: center; margin-top: 0;'>{max_val:.2f}</p>", unsafe_allow_html=True)

    col4.markdown('<p style="font-size: 16px; text-align: center; margin-bottom: 2px;">CI Lower</p>', unsafe_allow_html=True)
    col4.markdown(f"<p style='font-size: 18px; text-align: center; margin-top: 0; color: red;'>{ci_lower:.2f}</p>", unsafe_allow_html=True)

    col5.markdown('<p style="font-size: 16px; text-align: center; margin-bottom: 2px;">CI Upper</p>', unsafe_allow_html=True)
    col5.markdown(f"<p style='font-size: 18px; text-align: center; margin-top: 0; color: green;'>{ci_upper:.2f}</p>", unsafe_allow_html=True)

    col6.markdown('<p style="font-size: 16px; text-align: center; margin-bottom: 2px;">Std. Dev.</p>', unsafe_allow_html=True)
    col6.markdown(f"<p style='font-size: 18px; text-align: center; margin-top: 0;'>{std_dev:.2f}</p>", unsafe_allow_html=True)

    if not filtered_data.empty:
        if sport == "NFL":
            text_type = 'Weeks'
        else:
            text_type = 'Games'
        plot_graph(filtered_data, stat, graph_spacing, player,text_type)
    else:
        st.warning("No data available for the selected player and stat.")


    bin_size = st.sidebar.number_input(
        "Enter bin size for grouping",
        min_value=1,
        max_value=int(filtered_data[stat].max()),
        value=1,  # Default bin size
        step=10
    )

    bin_ranges = pd.cut(filtered_data[stat], bins=range(0, int(filtered_data[stat].max()) + bin_size, bin_size),right=False)
    bin_counts = bin_ranges.value_counts(sort=False)
    plot_data = (filtered_data.assign(Ranges=bin_ranges)
        .groupby('Ranges', as_index=False)
        .size()
        .rename(columns={"size": "Weeks"})
    )
    plot_data['Range'] = plot_data['Ranges'].apply(lambda x: f"{int(x.left)}-{int(x.right - 1)}")
    plot_data = plot_data[['Range', 'Weeks']]

    fig, ax = plt.subplots(figsize=(6, 2))  # Ultra-compact size
    ax.barh(plot_data['Range'], plot_data['Weeks'], color="skyblue", edgecolor="black", height=0.4)
    ax.set_xlabel("# of Games", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_title(f"Distribution of {player} {stat}", fontsize=9, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Adjust layout for a smaller chart
    plt.tight_layout()
    st.pyplot(fig)