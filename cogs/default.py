import nextcord
from helpers.utils import load_cogs, load_config
from nextcord.ext import commands

config = load_config()


class Default(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        super().__init__()
        self.bot = bot
        self.logger = bot.logger

    @commands.Cog.listener()
    async def on_message(self, message: nextcord.Message):
        """Capturing All messages"""
        self.logger.info(f"Received message: {message}")

        # if message.author == self.bot.user or message.author.bot:
        #     return

    @nextcord.slash_command(guild_ids=[config["guild_id"]], description="Reload Cogs")
    async def reload_cogs(self, interaction: nextcord.Interaction):
        await interaction.response.defer()
        self.logger.info("Reloading Cogs")
        await interaction.followup.send(content="Reloading Cogs", delete_after=300)
        failed_extensions = await load_cogs(self.bot, config, reload=True)
        if failed_extensions:
            self.logger.error(f"Could not load all extensions: {failed_extensions}")
            for ext in failed_extensions:
                await interaction.followup.send(
                    content=f"Could not load extension: {ext}", delete_after=300
                )
        else:
            self.logger.info("Cogs Reloaded")
            await interaction.followup.send(content="Cogs Reloaded", delete_after=300)
        await self.bot.sync_application_commands(guild_id=config["guild_id"])


def setup(bot: commands.Bot) -> None:
    bot.add_cog(Default(bot))
