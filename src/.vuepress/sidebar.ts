import { sidebar } from "vuepress-theme-hope";

// 图标：https://theme-hope.vuejs.press/zh/guide/interface/icon.html#%E8%AE%BE%E7%BD%AE%E5%9B%BE%E6%A0%87
// https://fontawesome.com/search?m=free&o=r
export default sidebar({
  "/": [
    {
      text: "技术栈",
      icon: "material-symbols:stack",
      prefix: "techstack/",
      link: "techstack/README.md",
      collapsible: true,
      expanded: true,
      children: [
        {
          text: "软件框架",
          icon: "fa6-solid:cube",
          // collapsible: true,
          // expanded: true,
          children: ["PyTorch.md"]
        },
      ],
    },
    // icon: software, code

    // {
    //   text: "🏗️ 网站部署",
    //   icon: "",
    //   prefix: "deploy/",
    //   collapsible: true,
    //   children: [
    //     "Static.md",
    //     "CloudServices.md",
    //     "VPS.md",
    //     {
    //       text: "部署工具",
    //       icon: "fa6-brands:windows",
    //       collapsible: true,
    //       children: ["GitHub.md", "Cloudflare.md", "MySQL.md", "DNS.md"],
    //     },
    //   ],
    // },
    // {
    //   text: "🔡 代码编程",
    //   icon: "",
    //   prefix: "code/",
    //   collapsible: true,
    //   children: [
    //     "README.md",
    //     {
    //       text: "Basic",
    //       icon: "fa6-solid:cube",
    //       collapsible: true,
    //       children: ["Markdown.md", "Electron.md", "AutoHotkey.md", "Regex.md"],
    //     },
    //     {
    //       text: "FrondEnd",
    //       icon: "fa6-solid:object-group",
    //       collapsible: true,
    //       children: ["Vue.md", "HTML.md", "Javascript.md", "Python.md"],
    //     },
    //   ],
    // },
    // {
    //   text: "🛖 生活记录",
    //   icon: "",
    //   prefix: "family/",
    //   collapsible: true,
    //   children: "structure",
    // },
    // {
    //   text: "加密目录",
    //   icon: "material-symbols:encrypted",
    //   prefix: "encrypt/",
    //   collapsible: true,
    //   children: "structure",
    // },
    {
      text: "博客文章",
      icon: "fa6-solid:blog",
      prefix: "_posts/",
      // link: "/blog",
      collapsible: true,
      children: "structure",
    },
  ],
  // 专题区（独立侧边栏）
  // "/apps/topic/": "structure",
  // 如果你不想使用默认侧边栏，可以按照路径自行设置。但需要去掉下方配置中的注释，以避免博客和时间轴出现异常。_posts 目录可以不存在。
  /*"/_posts/": [
    {
      text: "博客文章",
      icon: "fa6-solid:feather-pointed",
      prefix: "",
      link: "/blog",
      collapsible: true,
      children: "structure",
    },
  ], */
});
