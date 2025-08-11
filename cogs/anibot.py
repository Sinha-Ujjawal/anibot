import nextcord
from nextcord.ext import commands

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.globals import set_debug, set_verbose

import duckdb
import pandas as pd

from glob import glob
import json
from pathlib import Path
from typing import Tuple, List
from functools import wraps, cached_property

from helpers.utils import load_config

set_debug(True)
set_verbose(True)


def pysqldf(query: str, **kwargs) -> pd.DataFrame:
    for k, v in kwargs.items():
        duckdb.register(k, v)
    return duckdb.query(query).to_df()


config = load_config()

TOP_N_ANIME_MODEL = "gemma3:4b"
SQL_MODEL = "gemma3:4b"


def load_anime_data(
    json_files: Tuple[str] = glob("./data/anime-dataset/data/anime/*.json"),
) -> pd.DataFrame:
    """
    Load Anime Data from https://github.com/meesvandongen/anime-dataset/tree/main repo
    Specifically from data/anime/*.json files

    Here I have cloned the repository inside of ./data/anime-dataset, so by default loading from there

    The json files have below keys, but however we will be loading only relevant keys
        id
        title
        main_picture
        alternative_titles
        start_date
        end_date
        synopsis
        mean
        rank
        popularity
        num_list_users
        num_scoring_users
        nsfw
        created_at
        updated_at
        media_type
        status
        genres
        num_episodes
        start_season
        source
        average_episode_duration
        rating
        studios
    """
    data = []
    for json_file in json_files:
        anime_data = json.loads(Path(json_file).read_text())
        if anime_data["nsfw"] == "gray":
            continue
        alternative_names = "|".join(anime_data["alternative_titles"]["synonyms"])
        genre_set = set(genre["name"] for genre in anime_data.get("genres", []))
        genres = "|".join(genre_set)
        studios = "|".join(studio["name"] for studio in anime_data.get("studios", []))
        if len({"Hentai", "Erotica", "Magical Sex Shift"} & genre_set):
            continue
        data.append(
            {
                "id": anime_data["id"],
                "title": anime_data["title"],
                "alternative_names": alternative_names,
                "media_type": anime_data["media_type"],
                "start_date": anime_data["start_date"],
                "end_date": anime_data.get("end_date"),
                "rating": anime_data.get("rating"),
                "rank": anime_data.get("rank"),
                "genres": genres,
                "studios": studios,
                "nsfw": anime_data["nsfw"],
                "status": anime_data["status"],
                "num_episodes": anime_data["num_episodes"],
                "average_episode_duration_in_secs": anime_data[
                    "average_episode_duration"
                ],
                "synopsis": anime_data["synopsis"],
            }
        )
    return pd.DataFrame(data)


class TopNQuery(BaseModel):
    top_n: int = Field(description="N in Top N anime")
    genres: List[str] = Field(description="List of genres", default=[], required=False)
    years: List[int] = Field(description="List of years", default=[], required=False)
    media_types: List[str] = Field(
        description="List of Media types", default=[], required=False
    )
    studios: List[str] = Field(
        description="List of studios", default=[], required=False
    )


class AniBot(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        super().__init__()
        self.bot = bot
        self.logger = bot.logger

        self.anime_df = load_anime_data()

        # Setup Top N Query Model
        self.top_n_llm = ChatOllama(model=TOP_N_ANIME_MODEL, temperature=0)
        self.top_n_animes_model_prompt = PromptTemplate.from_template(
            (
                Path(__file__).parent
                / "anibot_structured_output_top_n_animes_model_prompt.txt"
            ).read_text()
        )
        self.top_n_animes_parser = PydanticOutputParser(pydantic_object=TopNQuery)
        self.top_n_animes_chain: Runnable = (
            self.top_n_animes_model_prompt | self.top_n_llm | self.top_n_animes_parser
        )

        # Setup SQL Model
        self.sql_llm = ChatOllama(model=SQL_MODEL, temperature=0)
        self.sql_prompt = PromptTemplate.from_template(
            (Path(__file__).parent / "anibot_sql_prompt.txt").read_text()
        )
        self.sql_chain = self.sql_prompt | self.sql_llm

    @cached_property
    def count_by_genres(self) -> List[Tuple[str, int]]:
        anime_df = self.anime_df.copy()
        anime_df["genres"] = anime_df["genres"].str.split("|").apply(tuple)
        count_map = {}
        for genres, cnt in anime_df.groupby("genres").size().reset_index().values:
            for genre in genres:
                if genre.strip() == "":
                    continue
                count_map[genre] = count_map.get(genre, 0) + cnt
        return sorted(count_map.items(), key=lambda kv: kv[1], reverse=True)

    @cached_property
    def unique_genres(self) -> List[str]:
        return [genre for genre, _ in self.count_by_genres]

    @cached_property
    def count_by_studios(self) -> List[Tuple[str, int]]:
        anime_df = self.anime_df.copy()
        anime_df["studios"] = anime_df["studios"].str.split("|").apply(tuple)
        count_map = {}
        for studios, cnt in anime_df.groupby("studios").size().reset_index().values:
            for studio in studios:
                if studio.strip() == "":
                    continue
                count_map[studio] = count_map.get(studio, 0) + cnt
        return sorted(count_map.items(), key=lambda kv: kv[1], reverse=True)

    @cached_property
    def unique_studios(self) -> List[str]:
        return [studio for studio, _ in self.count_by_studios]

    @cached_property
    def unique_media_types(self) -> List[str]:
        return sorted(
            self.anime_df[~self.anime_df["media_type"].isnull()]["media_type"].unique()
        )

    @staticmethod
    def respond_on_error(handler):
        @wraps(handler)
        async def _inner(self, interaction: nextcord.Interaction, *args, **kwargs):
            try:
                await handler(self, interaction, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error occurred: {e}")
                await self.followup(
                    interaction, "Some error occurred!", delete_after=300
                )

        return _inner

    @staticmethod
    async def followup(
        interaction: nextcord.Interaction,
        content: str,
        *rest_args,
        **rest_kwargs,
    ):
        if len(content) > 2000:
            content = "Response is greater than 2000 characters. Sorry, that is the max allowed characters ¯\_(ツ)_/¯"
        await interaction.followup.send(content=content, *rest_args, **rest_kwargs)

    @nextcord.slash_command(
        name="anibot_unique_genres",
        guild_ids=[config["guild_id"]],
        description="Returns list of unique anime genres, and their corresponding counts",
    )
    @respond_on_error
    async def unique_genres_cmd(self, interaction: nextcord.Interaction):
        await interaction.response.defer()
        content = "Unique Genres with their Counts:"
        for idx, (genre, cnt) in enumerate(self.count_by_genres, 1):
            content += f"\n{idx}. {genre}: {cnt}"
        await self.followup(interaction, content=content)

    @nextcord.slash_command(
        name="anibot_unique_studios",
        guild_ids=[config["guild_id"]],
        description="Returns list of unique anime studios, and their corresponding counts",
    )
    @respond_on_error
    async def unique_studios_cmd(
        self,
        interaction: nextcord.Interaction,
        start_at: int = nextcord.SlashOption(
            description="Start index (Note that this is 0 based)",
            required=False,
            default=0,
        ),
        count: int = nextcord.SlashOption(
            description="Max count of results to return",
            required=False,
            default=5,
        ),
    ):
        await interaction.response.defer()
        count_by_studios = self.count_by_studios[start_at : start_at + count]
        studios_or_studio = "studios" if len(count_by_studios) > 1 else "studio"
        content = f"Top {count} {studios_or_studio}, starting at {start_at} with their Counts:\n"
        for idx, (studio, cnt) in enumerate(count_by_studios):
            content += f"\n[{idx + start_at}] {studio}: {cnt}"
        await self.followup(interaction, content=content)

    @nextcord.slash_command(
        name="anibot_unique_media_types",
        guild_ids=[config["guild_id"]],
        description="Returns list of unique anime media types",
    )
    @respond_on_error
    async def unique_media_types_cmd(self, interaction: nextcord.Interaction):
        await interaction.response.defer()
        content = "Unique Media Types:"
        for idx, genre in enumerate(self.unique_media_types, 1):
            content += f"\n{idx}. {genre}"
        await self.followup(interaction, content=content)

    @staticmethod
    def anime_results_to_json(result_df: pd.DataFrame) -> str:
        result_df = result_df.copy()
        cols = [
            "id",
            "rank",
            "title",
            "start_date",
            "end_date",
            "genres",
            "media_type",
            "rating",
            "studios",
        ]
        if len(result_df) > 5:
            cols = ["id", "title", "rank", "genres", "studios"]
        return result_df[cols].to_json(orient="records", indent=1)

    def top_n_animes(
        self,
        *,
        n: int,
        genres: List[str],
        years: List[int],
        media_types: List[str],
        studios: List[str],
    ) -> str:
        anime_df = self.anime_df
        anime_df = anime_df[~anime_df["rank"].isnull()]

        n = max(n, 1)

        genre_set = set(genre.strip() for genre in genres if genre.strip())
        genres_combined = ", ".join(sorted(genre_set))
        if genre_set:
            anime_df = anime_df[
                anime_df["genres"]
                .str.split("|")
                .apply(
                    lambda xs: len(
                        set(map(lambda x: x.lower(), genre_set))
                        & set(map(lambda x: x.lower(), xs))
                    )
                    == 1
                )
            ]

        year_set = set(str(year).strip() for year in years if str(year).strip())
        years_combined = ", ".join(sorted(year_set))
        if year_set:
            anime_df = anime_df[
                anime_df["start_date"].apply(lambda dt: dt.split("-")[0] in year_set)
            ]

        media_type_set = set(
            media_type.strip() for media_type in media_types if media_type.strip()
        )
        media_types_comined = ", ".join(media_type_set)
        if media_type_set:
            anime_df = anime_df[
                anime_df["media_type"]
                .str.lower()
                .isin(map(lambda x: x.lower(), media_type_set))
            ]

        studio_set = set(studio.strip() for studio in studios if studio.strip())
        studios_combined = ", ".join(sorted(studio_set))
        if studio_set:
            anime_df = anime_df[
                anime_df["studios"]
                .str.split("|")
                .apply(
                    lambda xs: len(
                        set(map(lambda x: x.lower(), studio_set))
                        & set(map(lambda x: x.lower(), xs))
                    )
                    == 1
                )
            ]

        top_str = f"Top {n}" if n > 1 else "Top"
        anime_df = anime_df.sort_values(by="rank").iloc[:n]

        animes_or_anime = "animes" if n > 1 else "anime"
        genre_comment = ""
        year_comment = ""
        media_type_comment = ""
        studio_comment = ""
        if genre_set:
            genres_or_genre = "genres" if len(genre_set) > 1 else "genre"
            genre_comment = f" in {genres_combined} {genres_or_genre}"
        if year_set:
            years_or_year = "years" if len(year_set) > 1 else "year"
            year_comment = f" in {years_or_year} {years_combined}"
        if media_type_set:
            media_types_or_media_type = (
                "media_types" if len(media_type_set) > 1 else "media_type"
            )
            media_type_comment = (
                f" with {media_types_or_media_type} {media_types_comined}"
            )
        if studio_set:
            studios_or_studio = "studios" if len(studio_set) > 1 else "studio"
            studio_comment = f", produced by {studios_combined} {studios_or_studio}"
        return (
            f"{top_str} {animes_or_anime}{genre_comment}{year_comment}{media_type_comment}{studio_comment}:\n```\n"
            + self.anime_results_to_json(anime_df)
            + "```"
        )

    @nextcord.slash_command(
        name="anibot_top_n_animes",
        guild_ids=[config["guild_id"]],
        description="Returns Top N animes based on specific filters",
    )
    @respond_on_error
    async def top_n_animes_cmd(
        self,
        interaction: nextcord.Interaction,
        n: int,
        genres: str = nextcord.SlashOption(
            description="Enter genres separated by commas (e.g., Action, Comedy, Drama)",
            required=False,
            default="",
        ),
        years: str = nextcord.SlashOption(
            description="Enter years separated by commas (e.g., 2023, 2024, 2025)",
            required=False,
            default="",
        ),
        media_types: str = nextcord.SlashOption(
            description="Enter media_types separated by commas (e.g., tv, movie, music)",
            required=False,
            default="",
        ),
        studios: str = nextcord.SlashOption(
            description="Enter studios separated by '|' (e.g., Toei Animation | Sunrise | J.C.Staff)",
            required=False,
            default="",
        ),
    ):
        await interaction.response.defer()
        result = self.top_n_animes(
            n=n,
            genres=genres.split(","),
            years=years.split(","),
            media_types=media_types.split(","),
            studios=studios.split("|"),
        )
        await self.followup(interaction, content=result)

    @nextcord.slash_command(
        name="anibot_synopsis",
        guild_ids=[config["guild_id"]],
        description="Returns synopsis of a given anime id",
    )
    @respond_on_error
    async def synopsis_cmd(self, interaction: nextcord.Interaction, anime_id: int):
        await interaction.response.defer()
        anime_df = self.anime_df
        anime_df = anime_df[anime_df["id"] == anime_id]
        if len(anime_df):
            anime_title = anime_df["title"].iloc[0]
            synopsis = anime_df["synopsis"].iloc[0]
            result = f"Synopsis of anime `{anime_title}`, with id: `{anime_id}`"
            result += "\n```" + synopsis + "\n```"
        else:
            result = "Anime not found!"
        await self.followup(interaction, content=result)

    @nextcord.slash_command(
        name="anibot_top_n_animes_llm",
        guild_ids=[config["guild_id"]],
        description="Returns Top N animes based on specific filters generated from llm",
    )
    @respond_on_error
    async def top_n_animes_llm_cmd(self, interaction: nextcord.Interaction, query: str):
        await interaction.response.defer()
        query = query.strip()
        try:
            top_n_animes_query = self.top_n_animes_chain.invoke(
                {
                    "input": query,
                    "unique_genres": self.unique_genres,
                    "unique_media_types": self.unique_media_types,
                    "unique_studios": self.unique_studios,
                }
            )
            try:
                result = self.top_n_animes(
                    n=top_n_animes_query.top_n,
                    genres=top_n_animes_query.genres,
                    years=top_n_animes_query.years,
                    media_types=top_n_animes_query.media_types,
                    studios=top_n_animes_query.studios,
                )
            except Exception as e:
                self.logger.error(f"Error occurred while generating result: {e}")
                result = "`Unable to generate result`"
            response = f"""Top N Anime Query Output (from model: {TOP_N_ANIME_MODEL}):
```
{top_n_animes_query}
```
{result}
"""
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            response = "Unable to extract info from query"
        self.logger.info(f"{top_n_animes_query=}")
        content = f"""Your Query:
```
{query}
```
{response}
"""
        await self.followup(interaction, content=content)

    @nextcord.slash_command(
        name="anibot_sql_llm",
        guild_ids=[config["guild_id"]],
        description="Query the Anime Dataset using Natural Language",
    )
    @respond_on_error
    async def sql_llm_cmd(
        self, interaction: nextcord.Interaction, query: str, run_query: bool = True
    ):
        await interaction.response.defer()
        generated_sql = self.sql_chain.invoke(
            {
                "input": query,
                "unique_genres": self.unique_genres,
                "unique_media_types": self.unique_media_types,
                "unique_studios": self.unique_studios,
            }
        ).content
        if "```sql" in generated_sql:
            generated_sql = generated_sql.replace("```sql", "").replace("```", "")
        results = ""
        if run_query:
            anime_df = self.anime_df
            results_df = pysqldf(generated_sql, anime_df=anime_df)
            results_df_as_json = results_df.to_json(orient="records", indent=1)
            results = f"""Result:
```
{results_df_as_json}
```
"""
        content = f"""Your Query:
```
{query}
```
Generated SQL (from model: {SQL_MODEL}):
```
{generated_sql}
```
{results}
"""
        await self.followup(interaction, content=content)


def setup(bot: commands.Bot) -> None:
    bot.add_cog(AniBot(bot))
