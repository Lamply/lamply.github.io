import SocialLink from "./components/SocialLink.js";
import { defineClientConfig } from "vuepress/client";
import { setupTransparentNavbar } from "vuepress-theme-hope/presets/transparentNavbar.js";
import "./fonts/AlibabaPuHuiTi-3-55-Regular.ttf"
import "./fonts/AlibabaPuHuiTi-3-75-SemiBold.ttf"
import "./fonts/AlibabaPuHuiTi-3-85-Bold.ttf"

export default defineClientConfig({
  // setup: () => {
  //   setupTransparentNavbar({
  //     // Options (all optional)
  //     type: "all",           // "homepage" | "blog-homepage" | "all"
  //     threshold: 50,              // Scroll threshold in pixels before becoming opaque
  //     // light: "#ffffff",        // Text color in light mode (optional)
  //     // dark: "#000000",         // Text color in dark mode (optional)
  //   });
  // },
  enhance: ({ app }) => {
    app.component("SocialLink", SocialLink);
  },
});
