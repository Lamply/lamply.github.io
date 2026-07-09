import { viteBundler } from "@vuepress/bundler-vite";
import { defineUserConfig } from "vuepress";
import Font from "vite-plugin-font"
// 谷歌分析：启用时取消本行注释，并填回下方 plugins 里你自己的 GA4 ID
// import { googleAnalyticsPlugin } from "@vuepress/plugin-google-analytics";

// Shim navigator for SSR
if (typeof navigator === "undefined") {
  // @ts-ignore
  global.navigator = { userAgent: "node", platform: "node" };
}

import theme from "./theme.js";
import { faqSchemaPlugin } from "./plugins/faq-schema.js";

export default defineUserConfig({
  // 网站路径默认为主域名。如果网站部署在子路径下，比如 xxx.com/yyy，那么 base 应该被设置为 "/yyy/"
  base: "/",

  // 网站语言，默认为中文
  lang: "zh-CN",
  // 网站标题
  title: "Lamply Pages",
  // 网站描述
  description: "记录理性地觉察。",

  theme,
  // 是否开启页面预拉取，如果服务器宽带足够，可改为 true，会提升其他页面加载速度
  shouldPrefetch: false,

  // 修改页面模板，https://github.com/vuepress-theme-hope/vuepress-theme-hope/blob/main/packages/theme/templates/index.build.html
  // 配置参考：https://vuepress.github.io/zh/reference/theme-api.html#templatebuild
  templateBuild: "./src/.vuepress/templateBuild.html",

  // 禁止文件夹生成静态文件，参考 [VuePress 文档]（https://v2.vuepress.vuejs.org/zh/guide/page.html#routing）
  pagePatterns: ["**/*.md", "!_temp", "!reading", "!.vuepress", "!node_modules"],

  plugins: [
    // 谷歌分析：填回你自己的 GA4 ID 后，取消本段与顶部 import 的注释即可启用
    // googleAnalyticsPlugin({
    //   id: "G-XXXXXXXXXX",
    // }),
    // FAQ 结构化数据（frontmatter faq -> FAQPage JSON-LD）
    faqSchemaPlugin(),
  ],
  bundler: viteBundler({
    viteOptions: {
      plugins: [
        Font.vite({
          // 只切分中文字体文件
          include: [/AlibabaPuHuiTi-3-.*\.ttf$/],
          css: {
            // 显式指定切分后生成的 css font-family 统一名称
            fontFamily: "Alibaba PuHuiTi 3.0",
          },
        }),
      ],
    },
  }),
});
