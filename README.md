# AniBot 🎌

A Discord bot for anime discovery and recommendations powered by AI and a comprehensive anime database.

## Features

- **AI-Powered Recommendations**: Natural language anime queries using local LLMs
- **Advanced Filtering**: Filter by genres, years, studios, and media types
- **SQL Queries**: Convert natural language to SQL for complex database searches
- **Content Safe**: Automatically filters NSFW and inappropriate content
- **Rich Data**: Synopsis, ratings, rankings, and detailed anime information

## Prerequisites

- Python 3.8+
- [GEMINI\_API\_KEY](https://ai.google.dev/gemini-api/docs/api-key)
- Discord Bot Token

## Quick Setup

1. **Clone with submodules**:
   ```bash
   git clone --recursive <your-repo-url>
   cd anibot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get [GEMINI_API_KEY](https://ai.google.dev/gemini-api/docs/api-key), and set `GOOGLE_API_KEY` env variable in your `.env` file**

4. **Configure bot**:
   - Create your Discord application and get bot token
   - Update configuration with your guild ID
   - Add bot to your Discord server

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/anibot_top_n_animes` | Filter anime manually | `/anibot_top_n_animes n:5 genres:Action,Drama` |
| `/anibot_top_n_animes_llm` | AI-powered recommendations | `/anibot_top_n_animes_llm query:romantic comedy from 2020s` |
| `/anibot_sql_llm` | Natural language to SQL | `/anibot_sql_llm query:highest rated Studio Ghibli movies` |
| `/anibot_synopsis` | Get anime synopsis | `/anibot_synopsis anime_id:12345` |
| `/anibot_unique_genres` | List all genres | `/anibot_unique_genres` |
| `/anibot_unique_studios` | List studios with counts | `/anibot_unique_studios count:10` |

## Usage Examples

**AI Recommendations**:
```
/anibot_top_n_animes_llm query:top 5 action anime from 2020
/anibot_top_n_animes_llm query:best psychological thriller series
```

**Manual Filtering**:
```
/anibot_top_n_animes n:10 genres:Romance,Comedy years:2018,2019,2020
```

**SQL Queries**:
```
/anibot_sql_llm query:anime with more than 50 episodes and rating above 8
```

## Architecture

- **Data Source**: [anime-dataset](https://github.com/meesvandongen/anime-dataset) (included as submodule)
- **AI Models**: LangChain + Google Gemma Models for natural language processing
- **Database**: DuckDB for fast in-memory SQL operations
- **Discord**: Nextcord library for bot functionality

## Data

The bot uses a curated anime dataset with information on:
- Titles and alternative names
- Genres, studios, and media types
- Rankings, ratings, and episode counts
- Air dates and synopses

## Development

To extend the bot:
1. Add new slash commands in the `AniBot` class
2. Use the `@respond_on_error` decorator for error handling
3. Follow existing patterns for AI integration
4. Update prompt templates for new AI functionality

## Troubleshooting

- **Data loading**: Run `git submodule update --init` to fetch anime data

## Copyrights
Licensed under [@MIT](./LICENSE)
