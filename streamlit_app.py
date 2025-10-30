import streamlit as st
import requests
import pandas as pd


# FastAPI backend URL
GET_PLAYER_ID_API_URL = "https://api-974875114263.europe-west1.run.app/get_player_id"
SIMILAR_ALTERNATIVES_API_URL = "http://127.0.0.1:8000/find_similar_players"

st.title("MoneyBaller Player Similarity Search")

st.write("Select a player to find similar alternatives")

# Input box

player_name = st.text_input("Player Name", "")

if 'selected_player_id' in st.session_state:
    pass
else:
    st.info("No player selected yet.")

st.session_state
if player_name:
    try:
        response = requests.get(GET_PLAYER_ID_API_URL, params={"name": player_name})
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data)
                # Only show desired columns
                columns_to_show = [
                    'long_name', 'short_name', 'nationality_name', 'club_name', 'player_positions', 'player_id'
                ]

                # st.dataframe(df[columns_to_show])

            else:
                st.warning(f"No similar players found for: {player_name}")
        else:
            st.error(f"Error from API: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")

    # create a display label column (minimal extra column)
    df['display_label'] = df.apply(
        lambda r: f"{r.get('long_name') or r.get('short_name')} | {r.get('club_name','')} | {r.get('nationality_name','')} | {r.get('player_positions', '')}",
        axis=1
    )

    # clickable list with a button per row
    for i, row in df.iterrows():
        col1, col2 = st.columns([6,1])
        col1.write(row['display_label'])
        # unique key avoids collisions across reruns
        if col2.button("Select", key=f"select_{int(row['player_id'])}"):
            st.session_state['selected_player_id'] = int(row['player_id'])
            # st.experimental_rerun()  # optional: immediately re-run to reflect selection

    if 'selected_player_id' in st.session_state:
        st.success(f"Selected player_id: {st.session_state['selected_player_id']}")


# if 'selected_player_id' in st.session_state:
#     pass

if st.session_state.get('selected_player_id'):
    try:
        response = requests.get(SIMILAR_ALTERNATIVES_API_URL, params={"player_id": st.session_state['selected_player_id']})
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data)
                # Only show desired columns
                columns_to_show = [
                    "short_name", "player_positions", "overall", "pace",
                    "shooting", "passing", "dribbling", "defending",
                    "physic", "value_eur", "similarity"
                ]
                st.dataframe(df[columns_to_show])
            else:
                st.warning(f"No similar players found")
        else:
            st.error(f"Error from API: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
# else:
#     st.info("No player selected yet.")




import streamlit as st

html_code = """<!DOCTYPE html>
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />

    <!-- Stylesheets  -->
    <link href="https://fonts.googleapis.com/css?family=Comfortaa" rel="stylesheet" type="text/css">
  	<link href="https://fonts.googleapis.com/css?family=Istok+Web:400,700" rel="stylesheet" type="text/css" />

    <link rel="apple-touch-icon" sizes="57x57" href="/favicon/apple-icon-57x57.png">
				<link rel="manifest" href="/favicon/manifest.json">
		<meta name="msapplication-TileColor" content="#ffffff">
		<meta name="msapplication-TileImage" content="/favicon/ms-icon-144x144.png">
		<meta name="theme-color" content="#ffffff">
</head>
<body>
<nav class="navbar navbar-expand-lg bg-body-tertiary">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">Navbar</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarSupportedContent">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0">
        <li class="nav-item">
          <a class="nav-link active" aria-current="page" href="#">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Link</a>
        </li>
        <li class="nav-item dropdown">
          <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">
            Dropdown
          </a>
          <ul class="dropdown-menu">
            <li><a class="dropdown-item" href="#">Joueurs</a></li>
            <li><a class="dropdown-item" href="#">Similitudes</a></li>
            <li><hr class="dropdown-divider"></li>
            <li><a class="dropdown-item" href="#">Abonnement</a></li>
          </ul>
        </li>
        <li class="nav-item">
          <a class="nav-link disabled" aria-disabled="true">Autre</a>
        </li>
      </ul>
      <form class="d-flex" role="search">
        <input class="form-control me-2" type="search" placeholder="Search" aria-label="Search"/>
        <button class="btn btn-outline-success" type="submit">Recherche</button>
      </form>
    </div>
  </div>
</nav>
</body>

</html>"""

st.markdown(html_code, unsafe_allow_html=True)




