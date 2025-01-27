import pandas as pd
import numpy as np
import streamlit as st
from util_functions import calculate_statistics,  plot_graph
import matplotlib.pyplot as plt

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

sport = st.sidebar.selectbox("Select a Sport", ['NBA','NFL'])

if sport == 'NFL':
    @st.cache_data
    def load_nfl():
        file_path = "01-19-2025-nfl-season-player-feed.xlsx"
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
    @st.cache_data
    def load_nba():
        file_path = "01-26-2025-nba-season-player-feed.xlsx"
        df = pd.read_excel(file_path)
        df.columns = [col.replace("\n", "_").replace(" ", "_").upper() for col in df.columns]
        print(df.columns)
        df.rename(columns={'FG': 'Field Goals', 'FGA': 'Field Goal Attempts', '3P': 'Three-Point Field Goals',
                 '3PA': 'Three-Point Field Goal Attempts', 'FT': 'Free Throws','FTA': 'Free Throw Attempts',
                 'OR': 'Offensive Rebounds', 'DR': 'Defensive Rebounds', 'TOT': 'Total Rebounds',
                 'A': 'Assists', 'PF': 'Personal Fouls', 'ST': 'Steals', 'TO': 'Turnovers',
                 'BL': 'Blocks', 'PTS': 'Points','PLAYER__FULL_NAME': 'Player Name',
        }, inplace=True)
        return df
    df = load_nba()
    stat_options = ['Field Goals', 'Field Goal Attempts', 'Three-Point Field Goals', 'Three-Point Field Goal Attempts', 
                    'Free Throws', 'Free Throw Attempts', 'Offensive Rebounds', 'Defensive Rebounds', 
                    'Total Rebounds', 'Assists', 'Personal Fouls', 'Steals', 'Turnovers', 'Blocks', 
                    'Points'
                    ]
    df['WEEK'] = df.groupby('Player Name').cumcount() + 1




player = st.sidebar.selectbox("Select a Player", df["Player Name"].unique())
stat = st.sidebar.selectbox("Select a Metric", stat_options)
confidence_interval = st.sidebar.selectbox("Select Confidence Interval", options=[90, 95, 99], index=1 )
max_week = df['WEEK'].max()
num_games = st.sidebar.selectbox("Only Look at # of Last Games",  options=["Full Season"] + [str(i) for i in range(1, max_week + 1)])
num_options = np.arange(0, 2.1, 0.1)  
default_value = 0.5
default_index = list(num_options).index(default_value)
graph_spacing = st.sidebar.selectbox("Use to give spacing from points to text", num_options, index = default_index)

try:
    stat_metric = stat.split()[1]
except IndexError:
    stat_metric = stat

filtered_df = df[df["Player Name"] == player][["Player Name","WEEK", "DATE", stat]].dropna()
if num_games != "Full Season":
    num_games = int(num_games)
    player_week = filtered_df['WEEK'].max()
    last_weeks = filtered_df[filtered_df['WEEK'] >= (player_week - num_games + 1)]['WEEK']
    filtered_df = filtered_df[filtered_df['WEEK'].isin(last_weeks)]
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


mean, min_val, max_val, std_dev, ci_lower, ci_upper = calculate_statistics(filtered_df, stat,confidence_interval)

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

if not filtered_df.empty:
    if sport == "NFL":
        text_type = 'Weeks'
    else:
        text_type = 'Games'
    plot_graph(filtered_df, stat, graph_spacing, player,text_type)
else:
    st.warning("No data available for the selected player and stat.")


bin_size = st.sidebar.number_input(
    "Enter bin size for grouping",
    min_value=1,
    max_value=int(filtered_df[stat].max()),
    value=1,  # Default bin size
    step=10
)

bin_ranges = pd.cut(filtered_df[stat], bins=range(0, int(filtered_df[stat].max()) + bin_size, bin_size),right=False)
bin_counts = bin_ranges.value_counts(sort=False)
plot_data = (filtered_df.assign(Ranges=bin_ranges)
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